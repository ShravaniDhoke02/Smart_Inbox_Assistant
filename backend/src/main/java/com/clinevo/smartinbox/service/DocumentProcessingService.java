package com.clinevo.smartinbox.service;
import com.clinevo.smartinbox.dto.AIResponse;
import com.clinevo.smartinbox.entity.Attachment;
import com.clinevo.smartinbox.entity.ExtractedFact;
import com.clinevo.smartinbox.entity.Message;
import com.clinevo.smartinbox.repository.MessageRepository;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import java.time.LocalDateTime;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import com.fasterxml.jackson.databind.ObjectMapper;

/** A one-worker queue keeps OCR/AI calls outside the HTTP request path. */
@Service @RequiredArgsConstructor @Slf4j
public class DocumentProcessingService {
  private final AsyncProcessingQueue queue; private final MessageRepository messages; private final AuditService audit;
  private final ObjectMapper objectMapper = new ObjectMapper();
  private final RestTemplate http = new RestTemplate();
  @Value("${ai.service.url:http://localhost:8000/analyze}") private String aiUrl;
  @PostConstruct void startWorker(){ Executors.newSingleThreadExecutor().submit(this::run); }
  @Scheduled(fixedDelayString="${processing.recovery-delay-ms:30000}")
  void recoverPendingMessages(){ messages.findByStatusIn(List.of(Message.ProcessStatus.PENDING, Message.ProcessStatus.PROCESSING)).forEach(m -> queue.enqueue(m.getId())); }
  private void run(){ while(!Thread.currentThread().isInterrupted()) try { process(queue.dequeue()); } catch(InterruptedException e){Thread.currentThread().interrupt();} catch(Exception e){log.error("Queue worker error",e);} }
  private void process(Long id){
    Message m=messages.findWithAttachmentsById(id).orElse(null); if(m==null||m.getStatus()==Message.ProcessStatus.REVIEW_REQUIRED||m.getStatus()==Message.ProcessStatus.COMPLETED)return; restoreJson(m); long started=System.nanoTime(); m.setStatus(Message.ProcessStatus.PROCESSING); m.setProcessingProgress(10); m.setProcessingStage("Preparing document analysis"); m.setProcessingAttempts((m.getProcessingAttempts()==null?0:m.getProcessingAttempts())+1); m.setLastProcessingError(null); persistJson(m); messages.save(m);
    try {
      m.setProcessingProgress(25); m.setProcessingStage("Reading uploaded document"); messages.save(m);
      Map<String,Object> p=new LinkedHashMap<>(); p.put("message_id",String.valueOf(id)); p.put("subject",m.getSubject()); p.put("sender",m.getSender()); p.put("body",m.getBody());
      List<Map<String,String>> atts=new ArrayList<>(); if(m.getAttachments()!=null) for(Attachment a:m.getAttachments()) { Map<String,String> att=new LinkedHashMap<>();att.put("filename",a.getFilename());att.put("content_type",a.getFileType());if(a.getStoragePath()!=null&&Files.exists(java.nio.file.Path.of(a.getStoragePath())))att.put("base64_content",Base64.getEncoder().encodeToString(Files.readAllBytes(java.nio.file.Path.of(a.getStoragePath()))));atts.add(att); } p.put("attachments",atts);
      m.setProcessingProgress(50); m.setProcessingStage("AI is analysing document content"); messages.save(m);
      HttpHeaders h=new HttpHeaders(); h.setContentType(MediaType.APPLICATION_JSON);
      ResponseEntity<AIResponse> res=http.postForEntity(aiUrl,new HttpEntity<>(p,h),AIResponse.class);
      if(!res.getStatusCode().is2xxSuccessful()||res.getBody()==null)throw new IllegalStateException("AI service returned no analysis");
      m.setProcessingProgress(85); m.setProcessingStage("Saving AI results"); messages.save(m);
      apply(m,res.getBody()); m.setStatus(Message.ProcessStatus.REVIEW_REQUIRED); m.setProcessingProgress(100); m.setProcessingStage("Analysis complete — ready for review"); m.setProcessingDurationMs((System.nanoTime()-started)/1_000_000);m.setUpdatedAt(LocalDateTime.now()); persistJson(m); messages.save(m);
      audit.log("MESSAGE",id,"AI_ANALYZED","SYSTEM","Queued analysis completed in "+m.getProcessingDurationMs()+" ms");
    } catch(Exception e){
      log.warn("AI service unavailable; applying safe fallback to {}",id); m.setProcessingProgress(85);m.setProcessingStage("Saving safe fallback results");messages.save(m); apply(m,baseline(m));m.setStatus(Message.ProcessStatus.REVIEW_REQUIRED);m.setProcessingProgress(100);m.setProcessingStage("Analysis complete — ready for review");m.setProcessingDurationMs((System.nanoTime()-started)/1_000_000); persistJson(m); messages.save(m);
      m.setLastProcessingError(e.getMessage()); messages.save(m);
      audit.log("MESSAGE",id,"AI_FALLBACK","SYSTEM","Local baseline used: "+e.getClass().getSimpleName());
    }
  }
  private void apply(Message m,AIResponse r){
    m.setCategory(r.getCategory());m.setConfidenceScore(r.getConfidenceScore());m.setClassificationReason(r.getClassificationReason());m.setAiSummary(r.getAiSummary());m.setLanguage(r.getLanguage());m.setTranslatedText(r.getTranslatedText());m.setPdfTypes(r.getPdfTypes()==null?null:String.join(", ",r.getPdfTypes()));m.setImageNotes(r.getImageNotes()==null?null:String.join(" | ",r.getImageNotes()));
    try {
      m.setTables(r.getTables()==null?null:objectMapper.writeValueAsString(r.getTables()));
      m.setLiteratureScreening(r.getLiteratureScreening()==null?null:objectMapper.writeValueAsString(r.getLiteratureScreening()));
      m.setOcrConfidence(r.getOcrConfidence()==null?null:objectMapper.writeValueAsString(r.getOcrConfidence()));
      m.setSourceTrace(r.getSourceTrace()==null?null:objectMapper.writeValueAsString(r.getSourceTrace()));
    } catch(Exception e) {
      m.setTables(null); m.setLiteratureScreening(null); m.setSourceTrace(null);
      log.warn("Could not persist AI review metadata for message {}", m.getId(), e);
    }
    if(r.getPdfTypes()!=null&&m.getAttachments()!=null) for(int i=0;i<m.getAttachments().size()&&i<r.getPdfTypes().size();i++) m.getAttachments().get(i).setDetectedPdfType(r.getPdfTypes().get(i));
    Map<String, ExtractedFact> factsByField=new LinkedHashMap<>();
    if(r.getExtractedFacts()!=null) for(AIResponse.ExtractedFactDto d:r.getExtractedFacts()){ExtractedFact f=new ExtractedFact();f.setMessageId(m.getId());f.setFactGroup(d.getFactGroup());f.setFieldName(d.getFieldName());f.setFieldValue(d.getFieldValue());f.setConfidence(d.getConfidence());f.setSourceReference(d.getSourceReference());String key=(String.valueOf(d.getFactGroup()).trim()+"\u0000"+String.valueOf(d.getFieldName()).trim()).toLowerCase(Locale.ROOT);factsByField.merge(key,f,this::preferFact);}m.setExtractedFacts(new ArrayList<>(factsByField.values()));
  }
  private void restoreJson(Message m){try{if(m.getAttachmentsJson()!=null)m.setAttachments(objectMapper.readValue(m.getAttachmentsJson(),objectMapper.getTypeFactory().constructCollectionType(List.class,Attachment.class)));if(m.getExtractedFactsJson()!=null)m.setExtractedFacts(objectMapper.readValue(m.getExtractedFactsJson(),objectMapper.getTypeFactory().constructCollectionType(List.class,ExtractedFact.class)));}catch(Exception e){log.warn("Could not restore document JSON for {}",m.getId(),e);}}
  private void persistJson(Message m){try{m.setAttachmentsJson(objectMapper.writeValueAsString(m.getAttachments()==null?List.of():m.getAttachments()));m.setExtractedFactsJson(objectMapper.writeValueAsString(m.getExtractedFacts()==null?List.of():m.getExtractedFacts()));}catch(Exception e){throw new IllegalStateException("Could not persist document JSON",e);}}
  private ExtractedFact preferFact(ExtractedFact existing,ExtractedFact candidate){return isFallback(existing)&&!isFallback(candidate)?candidate:existing.getConfidence()==null||candidate.getConfidence()!=null&&candidate.getConfidence()>existing.getConfidence()?candidate:existing;}
  private boolean isFallback(ExtractedFact fact){return "Not stated".equalsIgnoreCase(String.valueOf(fact.getFieldValue()).trim())||String.valueOf(fact.getSourceReference()).toLowerCase(Locale.ROOT).contains("not stated in email body");}
  /** Keeps essential values visible when the optional Python/OCR service is down. */
  private List<AIResponse.ExtractedFactDto> extractBaselineFacts(String text){
    List<AIResponse.ExtractedFactDto> out=new ArrayList<>();
    addBaselineMatch(out,text,"Patient","Sex", "(?i)\\b(?:patient\\s*[:(]\\s*)?(female|male|woman|man)\\b",1);
    addBaselineMatch(out,text,"Patient","Age", "(?i)\\b(\\d{1,3})\\s+years?(?:\\s+old)?\\b|\\b(\\d{1,3})-year-old\\b",1);
    addBaselineMatch(out,text,"Patient","Name", "(?im)\\b(?:patient(?: name)?|name of patient)\\s*[:\\-]\\s*([A-Z][A-Za-z]+(?:\\s+[A-Z][A-Za-z.'-]+){1,3})",1);
    addBaselineMatch(out,text,"Product","Name", "(?i)\\b(?:started (?:on|taking)|taking|took|product\\s*[:\\-]|drug\\s*[:\\-])\\s+(?:the\\s+)?([A-Z][A-Za-z0-9-]*(?:\\s+XR)?(?:\\s*\\([A-Za-z0-9 -]+\\))?)",1);
    if(out.stream().noneMatch(f->"Product".equals(f.getFactGroup())&&"Name".equals(f.getFieldName()))){Matcher p=Pattern.compile("(?i)\\b(Cardi ozin|Cardiozin|Fevrolix|Neurotab\\s+XR)\\b").matcher(text);if(p.find())addBaseline(out,"Product","Name",p.group(1).replace(" ",""),.78);}
    addBaselineMatch(out,text,"Product","Disease / Indication", "(?i)\\b(?:for|indication\\s*[:\\-])\\s+([A-Za-z][A-Za-z -]{2,70}?)(?=\\s*(?:\\.|;|,\\s*(?:on|and|with)\\b|\\n|$))",1);
    addBaselineMatch(out,text,"Reaction","What", "(?i)\\b(?:developed|experienced|had)\\s+([^.!?\\n]+)",1);
    addBaselineMatch(out,text,"Reaction","Suspected Cause", "(?i)\\b(?:after|following|due to|possibly linked to|associated with)\\s+([^.!?\\n]{3,160})",1);
    return out;
  }
  private void addBaselineMatch(List<AIResponse.ExtractedFactDto> out,String text,String group,String field,String regex,int preferredGroup){Matcher match=Pattern.compile(regex).matcher(text);if(!match.find())return;String value=null;for(int i=preferredGroup;i<=match.groupCount();i++)if(match.group(i)!=null&&!match.group(i).isBlank()){value=match.group(i);break;}if(value==null)return;if("Sex".equals(field)){value=value.equalsIgnoreCase("woman")?"Female":value.equalsIgnoreCase("man")?"Male":value.substring(0,1).toUpperCase(Locale.ROOT)+value.substring(1).toLowerCase(Locale.ROOT);}addBaseline(out,group,field,value,.82);}
  private void addBaseline(List<AIResponse.ExtractedFactDto> out,String group,String field,String value,double confidence){AIResponse.ExtractedFactDto fact=new AIResponse.ExtractedFactDto();fact.setFactGroup(group);fact.setFieldName(field);fact.setFieldValue(value.trim());fact.setConfidence(confidence);fact.setSourceReference("Email body, local fallback matched text: "+value.trim());out.add(fact);}
  private AIResponse baseline(Message m){String t=(String.valueOf(m.getSubject())+" "+String.valueOf(m.getBody())).toLowerCase();AIResponse r=new AIResponse();boolean s=t.matches(".*(nausea|rash|adverse|reaction|hospital).*"),q=t.matches(".*(broken seal|damaged|contamination|defect|lot).*"),i=t.contains("?")||t.matches(".*(dose|interaction|dosing).*" );List<String> c=new ArrayList<>();if(s)c.add("Safety Report (ICSR)");if(q)c.add("Quality Complaint (PQC)");if(i&&!s&&!q)c.add("Info Request (MI)");if(c.isEmpty())c.add("Not Relevant");r.setCategory(String.join(", ",c));r.setConfidenceScore(.65);r.setClassificationReason("Safe local keyword baseline; reviewer verification required.");r.setAiSummary("Offline baseline classification: "+r.getCategory()+". The Python AI service was unavailable. No unsupported clinical facts were inferred. The message remains queued for human review. The reviewer should inspect the original email and any attached document. Facts not stated in the source remain Not stated. Classification confidence is provisional. Product, patient, reaction, and outcome details require confirmation. Any PDF should be checked page by page. The record must be reviewed before approval. This fallback does not make a clinical conclusion.");Map<String,Object> ocrConfidence=new LinkedHashMap<>();ocrConfidence.put("required",false);ocrConfidence.put("confidence",null);ocrConfidence.put("handwritingConfidence",null);ocrConfidence.put("method","OCR unavailable in offline fallback");r.setOcrConfidence(ocrConfidence);
    List<AIResponse.ExtractedFactDto> facts=new ArrayList<>();
    if(s){ for(String[] field: List.of(new String[]{"Patient","Age"},new String[]{"Patient","Sex"},new String[]{"Patient","Weight"},new String[]{"Patient","Height"},new String[]{"Patient","Relevant History"},new String[]{"Reporter","Role"},new String[]{"Reporter","Country"},new String[]{"Product","Name"},new String[]{"Product","Dose"},new String[]{"Product","Route"},new String[]{"Product","Therapy Start"},new String[]{"Product","Therapy Stop"},new String[]{"Reaction","What"},new String[]{"Reaction","Onset"},new String[]{"Reaction","Outcome"},new String[]{"Severity","Death"},new String[]{"Severity","Hospitalization"},new String[]{"Severity","Life-Threatening"},new String[]{"Narrative","Case Summary"})){ AIResponse.ExtractedFactDto fact=new AIResponse.ExtractedFactDto();fact.setFactGroup(field[0]);fact.setFieldName(field[1]);fact.setFieldValue("Not stated");fact.setConfidence(0.0);fact.setSourceReference("Not stated in email body or PDF attachment (offline fallback)");facts.add(fact); } }
    facts.addAll(extractBaselineFacts(String.valueOf(m.getSubject())+"\n"+String.valueOf(m.getBody())));
    r.setExtractedFacts(facts);return r;}
}
