package com.clinevo.smartinbox.service;

import com.clinevo.smartinbox.entity.Attachment;
import com.clinevo.smartinbox.entity.Message;
import com.clinevo.smartinbox.repository.MessageRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.mail.Folder;
import jakarta.mail.Multipart;
import jakarta.mail.Part;
import jakarta.mail.Session;
import jakarta.mail.Store;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Properties;

@Service
@RequiredArgsConstructor
@Slf4j
public class EmailIngestionService {

    private final MessageRepository messageRepository;
    private final AsyncProcessingQueue queue;
    private final ObjectMapper objectMapper;

    @Value("${mail.imap.host}")
    private String host;

    @Value("${mail.imap.port}")
    private String port;

    @Value("${mail.imap.username}")
    private String username;

    @Value("${mail.imap.password}")
    private String password;

    @Value("${mail.imap.protocol}")
    private String protocol;

    @Value("${mail.imap.folder:INBOX}")
    private String folderName;

    @Value("${mail.imap.enabled:false}")
    private boolean enabled;

    @Scheduled(fixedDelayString = "${mail.imap.poll-delay-ms:60000}")
    public void checkEmails() {
        if (!enabled) {
            return;
        }
        log.info("Checking for new emails...");
        Properties properties = new Properties();
        properties.put(String.format("mail.%s.host", protocol), host);
        properties.put(String.format("mail.%s.port", protocol), port);

        try {
            Session session = Session.getDefaultInstance(properties);
            Store store = session.getStore(protocol);
            store.connect(host, username, password);

            Folder inbox = store.getFolder(folderName);
            inbox.open(Folder.READ_WRITE);

            jakarta.mail.Message[] messages = inbox.getMessages();
            for (jakarta.mail.Message mail : messages) {
                if (!mail.isSet(jakarta.mail.Flags.Flag.SEEN)) {
                    processEmail(mail);
                    mail.setFlag(jakarta.mail.Flags.Flag.SEEN, true);
                }
            }
            inbox.close(false);
            store.close();
        } catch (Exception e) {
            log.error("Error reading emails: {}", e.getMessage());
        }
    }

    private void processEmail(jakarta.mail.Message mail) throws Exception {
        Message dbMessage = new Message();
        String[] messageIdHeaders = mail.getHeader("Message-ID");
        String messageId = messageIdHeaders != null && messageIdHeaders.length > 0 ? messageIdHeaders[0] : null;
        if (messageId != null && messageRepository.existsByExternalMessageId(messageId)) {
            log.info("Skipping already ingested email {}", messageId);
            return;
        }
        dbMessage.setExternalMessageId(messageId);
        dbMessage.setSubject(mail.getSubject());
        dbMessage.setSender(mail.getFrom()[0].toString());
        if (mail.getReceivedDate() != null) {
            dbMessage.setReceivedDate(mail.getReceivedDate().toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime());
        }
        dbMessage.setStatus(Message.ProcessStatus.PENDING);
        dbMessage.setAttachments(new ArrayList<>());

        StringBuilder bodyBuilder = new StringBuilder();

        if (mail.isMimeType("text/plain")) {
            bodyBuilder.append(mail.getContent().toString());
        } else if (mail.isMimeType("multipart/*")) {
            Multipart multipart = (Multipart) mail.getContent();
            for (int i = 0; i < multipart.getCount(); i++) {
                jakarta.mail.BodyPart bodyPart = multipart.getBodyPart(i);
                if (Part.ATTACHMENT.equalsIgnoreCase(bodyPart.getDisposition()) || bodyPart.getFileName() != null) {
                    // It's an attachment
                    String filename = bodyPart.getFileName();
                    if (filename != null && filename.toLowerCase().endsWith(".pdf")) {
                        Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), "smartinbox");
                        Files.createDirectories(tempDir);
                        File tempFile = tempDir.resolve(System.currentTimeMillis() + "_" + filename).toFile();
                        
                        try (InputStream is = bodyPart.getInputStream();
                             FileOutputStream fos = new FileOutputStream(tempFile)) {
                            byte[] buf = new byte[4096];
                            int bytesRead;
                            while ((bytesRead = is.read(buf)) != -1) {
                                fos.write(buf, 0, bytesRead);
                            }
                        }

                        Attachment attachment = new Attachment();
                        attachment.setFilename(filename);
                        attachment.setFileType("application/pdf");
                        attachment.setStoragePath(tempFile.getAbsolutePath());
                        dbMessage.getAttachments().add(attachment);
                    } else {
                        Attachment attachment = new Attachment();
                        attachment.setFilename(filename == null ? "unnamed-attachment" : filename);
                        attachment.setFileType(bodyPart.getContentType());
                        attachment.setDetectedPdfType("Logged only - non-PDF attachment");
                        dbMessage.getAttachments().add(attachment);
                        log.info("Logged non-PDF attachment {} without sending it to document processing", attachment.getFilename());
                    }
                } else if (bodyPart.isMimeType("text/plain")) {
                    bodyBuilder.append(bodyPart.getContent().toString());
                }
            }
        }
        dbMessage.setBody(bodyBuilder.toString());
        dbMessage.setAttachmentsJson(objectMapper.writeValueAsString(dbMessage.getAttachments()));
        dbMessage.setExtractedFactsJson("[]");
        Message savedMessage = messageRepository.save(dbMessage);
        log.info("Saved new message with ID: {}", savedMessage.getId());
        
        // Push to internal queue
        queue.enqueue(savedMessage.getId());
    }
}
