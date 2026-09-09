
import { Component, OnDestroy, OnInit } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { MessageService } from './services/message.service';

@Component({
  selector: 'app-root',
  template: `
    <section class="translation-banner" *ngIf="selected"><strong>Original language:</strong> {{selected.language || 'Unknown'}} <button class="text-button" type="button" (click)="translateSelected()" [disabled]="translating || refreshing">{{translating ? 'Translating...' : 'Translate email and PDF text'}}</button><span *ngIf="translation"><strong>English translation:</strong> {{translation}}</span></section>
    <div class="app-shell">
      <aside class="sidebar">
        <div class="brand"><span class="brand-mark"><span class="material-icons">inbox</span></span><span>ClinEvo AI</span></div>
        <nav class="nav-links" aria-label="Primary navigation">
          <button class="nav-item active" (click)="showQueue()"><span class="material-icons">check_circle_outline</span><span>Review Queue</span></button>
          <button class="nav-item" (click)="showHistory()"><span class="material-icons">history</span><span>History</span></button>
        </nav>
        <div class="sidebar-footer"><span class="status-dot"></span>AI Active</div>
      </aside>

      <main class="workspace">
        <header class="topbar">
          <div class="breadcrumb"><span>Workspace</span><span class="material-icons">chevron_right</span><strong>{{history ? 'History' : 'Review Queue'}}</strong></div>
          <div class="topbar-actions"><span class="secure-label"><span class="material-icons">verified_user</span>Secure workspace</span><button class="icon-button" title="Refresh" (click)="refresh()" [disabled]="refreshing" [attr.aria-label]="refreshing ? 'Refreshing' : 'Refresh'"><span class="material-icons">{{refreshing ? 'hourglass_empty' : 'refresh'}}</span></button><button class="avatar" title="Account">JD</button></div>
        </header>

        <section class="content" *ngIf="!selected; else reviewDetails">
          <div class="page-heading"><div><p class="eyebrow">{{history ? 'Processed messages' : 'Inbox operations'}}</p><h1>{{history ? 'History' : 'Review Queue'}}</h1><p class="subtitle">Review AI-classified healthcare correspondence before it moves forward.</p></div><span class="ai-pill"><span class="status-dot"></span>AI Active</span></div>
          <section class="queue-panel">
            <div class="panel-heading"><div><h2>{{history ? 'Message history' : 'Needs Human Review'}}</h2><p>{{messages.length ? messages.length + ' message' + (messages.length === 1 ? '' : 's') + ' waiting for review' : 'Your queue is clear'}}</p></div><div class="panel-actions"><button class="text-button" (click)="loadTestData()" [disabled]="loadingTestData"><span class="material-icons">science</span>Load Test Data</button><button class="text-button" (click)="clearTestData()"><span class="material-icons">delete_outline</span>Clear Test Data</button><label class="upload-button"><span class="material-icons">upload_file</span><span>Upload PDF</span><input type="file" accept="application/pdf" (change)="upload($event)"></label><button class="text-button" (click)="refresh()"><span class="material-icons">sync</span>Refresh</button></div></div>
            <div class="loader-panel" *ngIf="loadingTestData || loadingUpload" role="status" aria-live="polite"><div class="loader-heading"><strong>{{loadStatus}}</strong><span>{{loadProgress}}%</span></div><div class="progress-track"><div class="progress-value" [style.width.%]="loadProgress"></div></div></div>
            <div class="queue-table" *ngIf="messages.length; else emptyQueue">
              <div class="table-row table-header"><span>Date</span><span>Subject</span><span>AI Classification</span><span>Confidence</span><span>Action</span></div>
              <button class="table-row message-row" *ngFor="let item of messages" (click)="select(item)"><span class="date-cell">{{item.receivedAt || item.createdAt || 'Pending' | date:'MMM d, y'}}</span><span class="subject-cell"><strong>{{item.subject || 'Untitled message'}}</strong><small>{{item.sender || 'Healthcare correspondence'}}</small></span><span><span [class]="categoryClass(item)"><span class="material-icons pill-icon" *ngIf="isNotRelevant(item)">info</span>{{item.category || item.status || 'Pending'}}</span></span><span class="confidence">{{item.confidenceScore ? (item.confidenceScore * 100 | number:'1.0-0') + '%' : 'Pending'}}</span><span class="review-link">Review <span class="material-icons">arrow_forward</span></span></button>
            </div>
            <ng-template #emptyQueue><div class="empty-queue"><div class="empty-icon"><span class="material-icons">mail_outline</span></div><h3>No pending messages to review.</h3><p>New correspondence will appear here once it has been classified.</p></div></ng-template>
          </section>
          <p class="notice" *ngIf="notice"><span class="material-icons">info</span>{{notice}}</p>
        </section>

        <ng-template #reviewDetails>
          <section class="content review-content"><div class="page-heading"><div><p class="eyebrow">Message review</p><h1>Review details</h1><p class="subtitle">Inspect the source document and confirm the AI classification.</p></div><button class="back-button" (click)="selected = null"><span class="material-icons">arrow_back</span>Back to queue</button></div><div class="review-grid"><section class="review-card source"><h2>Source document</h2><iframe *ngIf="pdfUrl" [src]="pdfUrl"></iframe><p *ngIf="!pdfUrl" class="muted">No PDF is stored for this message.</p></section><section class="review-card details"><div class="not-relevant-card" *ngIf="isNotRelevant(selected)"><div class="not-relevant-header"><div class="not-relevant-title-wrap"><span class="material-icons not-relevant-icon">info_outline</span><div><h3>Data is Not Relevant to the Application</h3><p class="not-relevant-sub">Out of Pharmacovigilance Scope</p></div></div><span class="not-relevant-badge">Not Relevant</span></div><p class="not-relevant-message">The AI model analyzed this document/message and determined it is not relevant to the application's intake scope (ICSR Safety Reports, PQC Quality Complaints, or MI Information Requests). It appears to be general clinical literature without adverse events, administrative correspondence, or marketing material.</p><div class="not-relevant-reason-box" *ngIf="selected.classificationReason"><strong>Model Analysis:</strong> {{selected.classificationReason}}</div></div><h2>AI review</h2><label>Classification<input [(ngModel)]="selected.category"></label><label>Confidence<input type="number" min="0" max="1" step="0.01" [(ngModel)]="selected.confidenceScore"></label><label>Summary<textarea rows="6" [(ngModel)]="selected.aiSummary"></textarea></label><p class="reason"><strong>Reason:</strong> {{selected.classificationReason}}</p><p class="meta-line"><strong>Language:</strong> {{selected.language || 'Unknown'}} <span>·</span> <strong>PDF:</strong> {{selected.pdfTypes || 'None'}}</p><section class="evidence" *ngFor="let document of documentResults()"><h3>{{document.attachment}}</h3><p>{{document.pdfType}}</p><p>{{document.summary}}</p><p *ngFor="let ocr of ocrFor(document.attachment)"><strong>OCR:</strong> {{ocrLabel(ocr)}}</p></section><h3>Extracted facts</h3><div class="not-relevant-facts-notice" *ngIf="isNotRelevant(selected) && isOnlyNotStatedFacts()"><span class="material-icons">rule_folder</span><p>No safety or product quality facts extracted — Data is not relevant to the application.</p></div><article class="fact" *ngFor="let fact of (isNotRelevant(selected) && isOnlyNotStatedFacts() ? [] : sortedFacts())"><strong>{{fact.factGroup}} - {{fact.fieldName}}</strong><input [(ngModel)]="fact.fieldValue"><small>{{(fact.confidence || 0) * 100 | number:'1.0-0'}}% · {{fact.sourceReference}}</small></article><h3>Audit timeline</h3><ul class="audit"><li *ngFor="let event of audit">{{event.timestamp | date:'medium'}} - {{event.action}}: {{event.details}}</li></ul><div class="review-actions"><button class="approve-button" (click)="approve()"><span class="material-icons">check</span>Approve and save</button><button *ngIf="selected.status === 'REVIEW_REQUIRED' || selected.status === 'COMPLETED'" class="download-button" type="button" (click)="downloadJson()" [disabled]="downloadingJson"><span class="material-icons">download</span>{{downloadingJson ? 'Preparing JSON...' : 'Download JSON'}}</button></div></section></div></section>
        </ng-template>
      </main>
    </div>
  `,
  styles: [`
    :host { display: flex; flex-direction: column; height: 100vh; min-height: 0; color: #223047; overflow: hidden; }
    .app-shell { flex: 1 1 auto; min-height: 0; display: flex; background: #f7f9fc; overflow: hidden; }
    .sidebar { width: 244px; flex: 0 0 244px; display: flex; flex-direction: column; background: #17243a; color: #dce5f4; padding: 28px 16px 22px; box-sizing: border-box; }
    .brand { display: flex; align-items: center; gap: 11px; padding: 0 14px 38px; color: #fff; font-size: 19px; font-weight: 700; letter-spacing: -.2px; }
    .brand-mark { width: 28px; height: 28px; display: grid; place-items: center; border-radius: 7px; background: #3c74d8; color: #fff; }.brand-mark .material-icons { font-size: 18px; }
    .nav-links { display: grid; gap: 8px; }.nav-item { border: 0; display: flex; align-items: center; gap: 13px; width: 100%; padding: 13px 14px; border-radius: 7px; background: transparent; color: #9eabc1; font: inherit; font-size: 14px; text-align: left; cursor: pointer; }.nav-item:hover, .nav-item.active { background: #273a5a; color: #fff; }.nav-item.active { box-shadow: inset 3px 0 #5b8ff1; }.nav-item .material-icons { font-size: 20px; }
    .sidebar-footer { margin-top: auto; display: flex; align-items: center; gap: 8px; padding: 16px 14px 0; color: #94a5bf; font-size: 12px; border-top: 1px solid #2b3a53; }.status-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #45c88a; box-shadow: 0 0 0 3px rgba(69,200,138,.13); }
    .workspace { flex: 1; min-width: 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }.topbar { flex: 0 0 70px; min-height: 70px; display: flex; align-items: center; justify-content: space-between; padding: 0 42px; border-bottom: 1px solid #e5eaf1; background: rgba(255,255,255,.76); box-sizing: border-box; }.breadcrumb, .topbar-actions { display: flex; align-items: center; gap: 9px; }.breadcrumb { color: #8490a2; font-size: 13px; }.breadcrumb strong { color: #26354b; font-weight: 600; }.breadcrumb .material-icons { font-size: 17px; color: #b5bdc9; }.topbar-actions { gap: 18px; }.secure-label { display: flex; align-items: center; gap: 6px; color: #8995a8; font-size: 12px; }.secure-label .material-icons { font-size: 16px; color: #5394d9; }.icon-button, .avatar { border: 0; cursor: pointer; }.icon-button { background: transparent; color: #78879c; }.icon-button .material-icons { font-size: 20px; }.avatar { width: 31px; height: 31px; border-radius: 50%; background: #d7e5f7; color: #3b6ca5; font-size: 11px; font-weight: 700; }
    .content { width: 100%; max-width: 1260px; min-height: 0; margin: 0 auto; padding: 42px 48px 60px; box-sizing: border-box; }.content:not(.review-content) { overflow-y: auto; }.page-heading { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 30px; }.eyebrow { margin: 0 0 8px; color: #5e83b9; font-size: 11px; font-weight: 700; letter-spacing: 1.3px; text-transform: uppercase; }.page-heading h1 { margin: 0; color: #1d2d43; font-size: 29px; letter-spacing: -.6px; }.subtitle { margin: 9px 0 0; color: #8793a5; font-size: 13px; }.ai-pill { display: flex; align-items: center; gap: 8px; color: #3e719f; background: #edf6ff; padding: 8px 12px; border-radius: 18px; font-size: 12px; }
    .queue-panel, .review-card { border: 1px solid #e2e8f0; border-radius: 9px; background: #fff; box-shadow: 0 5px 18px rgba(35,55,80,.035); }.panel-heading { display: flex; justify-content: space-between; align-items: center; padding: 28px 30px 22px; border-bottom: 1px solid #edf0f4; }.panel-heading h2, .review-card h2 { margin: 0; color: #304158; font-size: 17px; font-weight: 650; }.panel-heading p { margin: 6px 0 0; color: #98a3b1; font-size: 12px; }.panel-actions { display: flex; align-items: center; gap: 17px; }.upload-button, .text-button, .back-button, .approve-button { display: inline-flex; align-items: center; gap: 7px; cursor: pointer; font: inherit; }.upload-button { padding: 9px 13px; border: 1px solid #d6e0eb; border-radius: 5px; color: #4f6d91; background: #fff; font-size: 12px; }.upload-button input { display: none; }.upload-button .material-icons, .text-button .material-icons { font-size: 16px; }.text-button { border: 0; color: #6381a6; background: transparent; font-size: 12px; }
    .queue-table { width: 100%; }.table-row { display: grid; grid-template-columns: 1.05fr 2fr 1.45fr .9fr .8fr; align-items: center; gap: 18px; padding: 0 30px; }.table-header { min-height: 47px; color: #97a2b1; font-size: 10px; font-weight: 700; letter-spacing: .8px; text-transform: uppercase; }.message-row { min-height: 72px; width: 100%; border: 0; border-top: 1px solid #edf0f4; background: #fff; color: #53647b; text-align: left; cursor: pointer; font: inherit; font-size: 12px; }.message-row:hover { background: #f8fbff; }.subject-cell { display: grid; gap: 5px; }.subject-cell strong { color: #32435a; font-size: 13px; font-weight: 600; }.subject-cell small { color: #9ba6b4; font-size: 11px; }.date-cell { color: #8795a7; }
    .classification { display: inline-block; padding: 5px 9px; border-radius: 4px; background: #eef5ff; color: #5478ad; font-size: 11px; }
    .classification.not-relevant { background: #fff4e5; color: #b76e00; border: 1px solid #ffe3b3; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; }
    .classification.icsr { background: #ebf3ff; color: #2b6cb0; font-weight: 600; }
    .classification.pqc { background: #fdf2f2; color: #9b2c2c; font-weight: 600; }
    .classification.mi { background: #e6fffa; color: #234e52; font-weight: 600; }
    .pill-icon { font-size: 14px; color: #b76e00; }
    .not-relevant-card { border: 1px solid #fbd38d; border-radius: 9px; background: linear-gradient(135deg, #fffaf0 0%, #ffffff 100%); padding: 18px 20px; box-shadow: 0 4px 12px rgba(221,107,32,0.08); margin-bottom: 16px; }
    .not-relevant-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px; }
    .not-relevant-title-wrap { display: flex; align-items: center; gap: 12px; }
    .not-relevant-icon { font-size: 24px; color: #dd6b20; background: #feebc8; padding: 6px; border-radius: 50%; }
    .not-relevant-header h3 { margin: 0; color: #9c4221; font-size: 16px; font-weight: 700; }
    .not-relevant-sub { margin: 2px 0 0; color: #c05621; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .not-relevant-badge { padding: 4px 10px; border-radius: 12px; background: #dd6b20; color: #ffffff; font-size: 11px; font-weight: 700; }
    .not-relevant-message { margin: 0 0 12px; color: #7b341e; font-size: 13px; line-height: 1.5; }
    .not-relevant-reason-box { padding: 10px 14px; border-radius: 6px; background: rgba(254,235,200,0.6); border-left: 3px solid #dd6b20; color: #7b341e; font-size: 12px; line-height: 1.45; }
    .not-relevant-facts-notice { display: flex; align-items: center; gap: 10px; padding: 14px 16px; border-radius: 7px; background: #f7fafc; border: 1px dashed #cbd5e0; color: #718096; font-size: 13px; margin-top: 8px; }
    .not-relevant-facts-notice .material-icons { color: #a0aec0; font-size: 20px; }
    .not-relevant-facts-notice p { margin: 0; }
    .confidence { color: #4e7b9d; font-weight: 650; }.review-link { display: inline-flex; align-items: center; gap: 3px; color: #4d82c5; font-weight: 600; }.review-link .material-icons { font-size: 15px; }
    .empty-queue { min-height: 290px; display: flex; flex-direction: column; align-items: center; justify-content: center; border-top: 1px solid #edf0f4; text-align: center; }.empty-icon { width: 44px; height: 44px; display: grid; place-items: center; margin-bottom: 16px; border: 1px solid #9fc1df; border-radius: 8px; color: #5a93c5; }.empty-icon .material-icons { font-size: 24px; }.empty-queue h3 { margin: 0; color: #58708b; font-size: 14px; font-weight: 500; }.empty-queue p { margin: 8px 0 0; color: #a2acb8; font-size: 12px; }.notice { display: flex; align-items: center; gap: 7px; color: #5d7793; font-size: 12px; }.notice .material-icons { font-size: 16px; }.loader-panel { padding: 16px 30px; border-bottom: 1px solid #edf0f4; background: #f8fbff; }.loader-heading { display: flex; justify-content: space-between; margin-bottom: 9px; color: #58708b; font-size: 12px; }.loader-heading span { color: #3977cf; font-weight: 700; }.progress-track { height: 7px; overflow: hidden; border-radius: 4px; background: #dfe9f4; }.progress-value { height: 100%; border-radius: 4px; background: #3977cf; transition: width .25s ease; }
    .review-content { flex: 1 1 auto; height: auto; padding-top: 34px; padding-bottom: 34px; overflow: hidden; }.back-button { border: 1px solid #d6e0eb; border-radius: 5px; padding: 9px 13px; background: #fff; color: #54779f; font-size: 12px; }.back-button .material-icons { font-size: 16px; }.review-grid { min-height: 0; height: calc(100% - 82px); display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(360px, .9fr); gap: 20px; }.review-card { min-height: 0; padding: 25px; box-sizing: border-box; }.review-card h2 { margin-bottom: 22px; }.source { display: flex; flex-direction: column; overflow: hidden; }.source iframe { width: 100%; flex: 1 1 auto; min-height: 0; border: 1px solid #e5eaf1; }.muted { color: #94a0af; font-size: 13px; }.details { display: grid; align-content: start; gap: 14px; overflow-y: auto; }.details label { display: grid; gap: 7px; color: #66778e; font-size: 12px; }.details input, .details textarea, .fact input { box-sizing: border-box; width: 100%; border: 1px solid #dce4ed; border-radius: 5px; padding: 9px 10px; color: #304158; font: inherit; font-size: 13px; }.details textarea { resize: vertical; }.details h3 { margin: 16px 0 0; color: #304158; font-size: 14px; }.reason, .meta-line, .evidence p { margin: 0; color: #77869a; font-size: 12px; line-height: 1.55; }.meta-line span { padding: 0 5px; color: #c2cad4; }.evidence { padding: 13px; border-left: 3px solid #8bb7e1; background: #f7faff; }.evidence h3 { margin: 0 0 4px; font-size: 12px; }.fact { display: grid; gap: 6px; padding: 11px 0; border-bottom: 1px solid #edf0f4; }.fact strong { color: #4b5f77; font-size: 12px; }.fact small { color: #9aa6b5; font-size: 11px; }.audit { margin: 0; padding-left: 18px; color: #78879a; font-size: 11px; line-height: 1.8; }.review-actions { display: flex; gap: 10px; flex-wrap: wrap; }.approve-button, .download-button { justify-content: center; border: 0; border-radius: 5px; padding: 11px 15px; color: #fff; font-size: 13px; font-weight: 600; cursor: pointer; }.approve-button { background: #3977cf; }.download-button { background: #2f8b6d; }.approve-button:disabled, .download-button:disabled { cursor: wait; opacity: .65; }.approve-button .material-icons, .download-button .material-icons { font-size: 17px; }
    @media (max-width: 820px) { .sidebar { width: 72px; flex-basis: 72px; padding: 22px 10px; }.brand { justify-content: center; padding: 0 0 30px; }.brand > span:last-child, .nav-item span:last-child, .sidebar-footer { display: none; }.nav-item { justify-content: center; padding: 13px 0; }.topbar { padding: 0 20px; }.secure-label { display: none; }.content { padding: 28px 20px 40px; }.table-row { grid-template-columns: 1fr 2fr 1fr; padding: 0 16px; }.table-row > :nth-child(4), .table-row > :nth-child(5) { display: none; }.review-grid { grid-template-columns: 1fr; }.panel-heading { align-items: flex-start; gap: 15px; flex-direction: column; } }
    @media (max-width: 500px) { .page-heading { gap: 18px; flex-direction: column; }.ai-pill { align-self: flex-start; }.topbar { min-height: 58px; }.breadcrumb span:first-child { display: none; }.review-card { padding: 18px; } }
  `]
})
export class AppComponent implements OnInit, OnDestroy {
  messages: any[] = [];
  selected: any;
  audit: any[] = [];
  history = false;
  notice = '';
  pdfUrl: SafeResourceUrl | null = null;
  translation = '';
  translating = false;
  loadingTestData = false;
  loadingUpload = false;
  downloadingJson = false;
  refreshing = false;
  loadProgress = 0;
  loadStatus = '';
  private loadMessageIds: number[] = [];
  private loadPollTimer?: number;
  private uploadPollTimer?: number;

  constructor(private api: MessageService, private sanitizer: DomSanitizer) {}

  ngOnInit() { this.showQueue(); }
  ngOnDestroy() { this.stopLoadPolling(); this.stopUploadPolling(); }
  showQueue() { this.history = false; this.selected = null; this.api.getPendingMessages().subscribe(items => this.messages = items); }
  showHistory() { this.history = true; this.selected = null; this.api.getHistory().subscribe(items => this.messages = items); }
  refresh() {
    if (this.refreshing) return;
    this.refreshing = true;
    const request = this.history ? this.api.getHistory() : this.api.getPendingMessages();
    request.subscribe({
      next: items => {
        this.messages = items;
        this.notice = 'Inbox refreshed.';
        this.refreshing = false;
      },
      error: error => {
        this.notice = this.errorMessage(error, 'Could not refresh the inbox.');
        this.refreshing = false;
      }
    });
  }
  select(item: any) { this.api.getMessage(item.id).subscribe(message => { this.selected = message; this.translation = ''; const pdf = (message.attachments || []).find((a: any) => (a.filename || '').toLowerCase().endsWith('.pdf')); this.pdfUrl = pdf ? this.sanitizer.bypassSecurityTrustResourceUrl(this.api.attachmentUrl(message.id, pdf.id)) : null; this.api.getAudit(message.id).subscribe(events => this.audit = events); }); }
  translateSelected() {
    if (!this.selected || this.translating) return;
    const source = String(this.selected.body || '').trim();
    if (!source) {
      this.translation = 'No source text is available for translation.';
      return;
    }
    this.translating = true;
    this.api.translate(source).subscribe({
      next: result => {
        this.translation = result?.translatedText || 'Translation unavailable.';
        this.translating = false;
      },
      error: error => {
        this.translation = this.errorMessage(error, 'Translation unavailable.');
        this.translating = false;
      }
    });
  }
  upload(event: Event) { const input = event.target as HTMLInputElement; const file = input.files?.[0]; if (!file) return; this.stopUploadPolling(); this.loadingUpload = true; this.loadProgress = 0; this.loadStatus = `Uploading ${file.name}...`; this.notice = ''; this.api.uploadPdf(file).subscribe({ next: result => { input.value = ''; this.loadStatus = 'Queued for AI analysis'; this.pollUploadStatus(Number(result.id)); }, error: error => { this.loadingUpload = false; this.notice = this.errorMessage(error, 'Upload failed.'); } }); }
  pollUploadStatus(messageId: number) { this.api.getProcessingStatus(messageId).subscribe({ next: status => { this.loadProgress = Number(status.progress || 0); this.loadStatus = status.stage || 'Processing document...'; if (status.terminal) { this.loadingUpload = false; this.stopUploadPolling(); this.refresh(); this.notice = status.status === 'FAILED' ? 'Processing failed. Please try again.' : 'Processing complete. Results have been refreshed.'; } else { this.uploadPollTimer = window.setTimeout(() => this.pollUploadStatus(messageId), 750); } }, error: error => { this.loadingUpload = false; this.notice = this.errorMessage(error, 'Unable to retrieve processing status.'); } }); }
  stopUploadPolling() { if (this.uploadPollTimer !== undefined) { window.clearTimeout(this.uploadPollTimer); this.uploadPollTimer = undefined; } }
  loadTestData() { this.stopLoadPolling(); this.loadingTestData = true; this.loadProgress = 0; this.loadStatus = 'Queuing test documents...'; this.notice = ''; this.api.loadTestData().subscribe({ next: result => { this.loadMessageIds = (result.messageIds || []).map((id: any) => Number(id)); this.notice = `${result.created || 0} test messages queued.`; this.pollLoadStatus(); }, error: error => { this.loadingTestData = false; this.notice = this.errorMessage(error, 'Could not load test data.'); } }); }
  pollLoadStatus() { this.api.getTestDataReport().subscribe({ next: report => { const records = (report.documents || []).filter((item: any) => this.loadMessageIds.includes(Number(item.id))); const finished = records.filter((item: any) => ['REVIEW_REQUIRED', 'COMPLETED', 'FAILED'].includes(item.status)).length; this.loadProgress = records.length ? Math.round(records.reduce((total: number, item: any) => total + Number(item.progress || 0), 0) / records.length) : 100; const active = records.find((item: any) => !['REVIEW_REQUIRED', 'COMPLETED', 'FAILED'].includes(item.status)); this.loadStatus = finished === this.loadMessageIds.length ? 'Processing complete. Refreshing results...' : active?.stage || `Processing ${finished} of ${this.loadMessageIds.length} documents...`; if (finished === this.loadMessageIds.length) { this.loadProgress = 100; this.loadingTestData = false; this.stopLoadPolling(); this.refresh(); } else { this.loadPollTimer = window.setTimeout(() => this.pollLoadStatus(), 1000); } }, error: () => { this.loadStatus = 'Waiting for processing status...'; this.loadPollTimer = window.setTimeout(() => this.pollLoadStatus(), 1500); } }); }
  stopLoadPolling() { if (this.loadPollTimer !== undefined) { window.clearTimeout(this.loadPollTimer); this.loadPollTimer = undefined; } }
  clearTestData() { this.notice = 'Clearing test data...'; this.api.clearTestData().subscribe({ next: result => { this.notice = result || 'Test data cleared.'; this.refresh(); }, error: error => this.notice = this.errorMessage(error, 'Could not clear test data.') }); }
  errorMessage(error: any, fallback: string): string { return error?.error && typeof error.error === 'string' ? error.error : `${fallback} Make sure the backend is running on port 8080.`; }
approve() {
  if (!this.selected) return;

  this.api.updateMessage(this.selected.id, this.selected).subscribe({
    next: saved => {
      this.notice = 'Record approved and saved.';

      // Switch to History so the approved record remains visible
      this.history = true;

      // Reload saved records from Oracle through the backend
      this.api.getHistory().subscribe(items => {
        this.messages = items;

        // Reload the complete saved record, PDF and audit
        this.select(saved);
      });
    },
    error: () => {
      this.notice = 'Could not save the review.';
    }
  });
}
  downloadJson() {
    if (!this.selected?.id || this.downloadingJson) return;
    this.downloadingJson = true;
    this.api.downloadJson(this.selected.id).subscribe({
      next: blob => {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `message-${this.selected.id}-extracted-data.json`;
        link.click();
        window.URL.revokeObjectURL(url);
        this.downloadingJson = false;
      },
      error: error => {
        this.notice = this.errorMessage(error, 'Could not download the JSON export.');
        this.downloadingJson = false;
      }
    });
  }
  trace(): any { try { return typeof this.selected?.sourceTrace === 'string' ? JSON.parse(this.selected.sourceTrace) : (this.selected?.sourceTrace || {}); } catch { return {}; } }
  sortedFacts(): any[] { const facts = this.selected?.extractedFacts || []; const hasPatientInitials = facts.some((fact: any) => String(fact.factGroup || '').toLowerCase() === 'patient' && String(fact.fieldName || '').toLowerCase() === 'initials' && String(fact.fieldValue || '').trim().toLowerCase() !== 'not stated'); const displayFacts = facts.filter((fact: any) => !(hasPatientInitials && String(fact.factGroup || '').toLowerCase() === 'patient' && String(fact.fieldName || '').toLowerCase() === 'name' && String(fact.fieldValue || '').trim().toLowerCase() === 'not stated')); const canonical = (fact: any) => { const group = String(fact.factGroup || '').toLowerCase().replace(/ information$/, '').trim(); let field = String(fact.fieldName || '').toLowerCase().trim(); field = field.replace(/^patient information\s*-?\s*/, '').replace(/^patient\s*-?\s*/, ''); if (['initials', 'id', 'name', 'patient initials', 'patient id', 'patient name'].includes(field)) field = 'identity'; if (field === 'age') field = 'age'; if (field === 'sex') field = 'sex'; return `${group}\u0000${field}`; }; const preferred = new Map<string, any>(); displayFacts.forEach((fact: any) => { const key = canonical(fact); const current = preferred.get(key); if (!current || Number(fact.confidence || 0) > Number(current.confidence || 0)) preferred.set(key, fact); }); const priority = ['patient\u0000identity', 'patient\u0000age', 'patient\u0000sex', 'patient\u0000weight', 'patient\u0000height', 'reaction\u0000what', 'reaction\u0000suspected cause', 'reaction\u0000description', 'adverse reaction\u0000reaction description', 'patient\u0000relevant history', 'product\u0000disease / indication', 'product\u0000indication', 'product\u0000name', 'product\u0000dose', 'product\u0000route', 'reaction\u0000onset', 'reaction\u0000outcome', 'severity\u0000death', 'severity\u0000hospitalization', 'severity\u0000life-threatening']; const rank = (fact: any) => { const index = priority.indexOf(canonical(fact)); return index === -1 ? priority.length : index; }; return [...preferred.values()].sort((left: any, right: any) => rank(left) - rank(right)); }
  documentResults(): any[] { const trace = this.trace(); const notes = this.imageNotes(); const tables = this.tables(); const literature = this.literature(); return (trace.attachmentSummaries || []).map((document: any) => { const documentNotes = notes.filter((note: string) => note.includes(document.attachment)); const documentTables = tables.filter((table: any) => table.attachment === document.attachment); const cases = literature?.cases?.filter((item: any) => item.sourceReference?.includes(document.attachment)) || []; const additions = [...documentNotes.map((note: string) => `Visual analysis: ${note}`), ...documentTables.map((table: any) => `Table: ${table.sourceReference || table.summary}`), ...(cases.length ? [`Literature screening isolated ${cases.length} patient case section(s).`] : [])]; return additions.length ? { ...document, summary: `${document.summary} ${additions.join(' ')}` } : document; }); }
  ocrFor(name: string): any[] { return (this.trace().ocrAssessments || []).filter((item: any) => item.attachment === name); }
  imageNotes(): string[] { const value = this.selected?.imageNotes || []; const parsed = this.parseJson(value); return Array.isArray(parsed) ? parsed.filter((note: any) => !!note) : String(value).split(' | ').filter((note: string) => !!note); }
  tables(): any[] { const value = this.parseJson(this.selected?.tables || []); return Array.isArray(value) ? value.filter((item: any) => typeof item === 'object') : []; }
  literature(): any { return this.parseJson(this.selected?.literatureScreening || {}); }
  parseJson(value: any): any { if (typeof value !== 'string') return value; try { return JSON.parse(value); } catch { return []; } }
  ocrLabel(ocr: any): string { const handwriting = ocr.handwritingConfidence === null || ocr.handwritingConfidence === undefined ? '' : `; handwriting ${Math.round((ocr.handwritingConfidence || 0) * 100)}%`; return ocr.required ? `${Math.round((ocr.confidence || 0) * 100)}% (${ocr.method}${handwriting})` : ocr.method; }
  isNotRelevant(item: any): boolean { if (!item) return false; const category = String(item.category || '').toLowerCase(); return category.includes('not relevant') || category.includes('irrelevant'); }
  categoryClass(item: any): string { if (this.isNotRelevant(item)) return 'classification not-relevant'; const cat = String(item?.category || '').toLowerCase(); if (cat.includes('safety') || cat.includes('icsr')) return 'classification icsr'; if (cat.includes('quality') || cat.includes('pqc')) return 'classification pqc'; if (cat.includes('info') || cat.includes('mi')) return 'classification mi'; return 'classification'; }
  isOnlyNotStatedFacts(): boolean { const facts = this.sortedFacts(); if (!facts || facts.length === 0) return true; return facts.every(f => String(f.fieldValue || '').toLowerCase() === 'not stated'); }
}
