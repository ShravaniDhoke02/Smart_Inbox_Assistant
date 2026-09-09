import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class MessageService {
  private apiUrl = '/api/messages';

  constructor(private http: HttpClient) { }

  getPendingMessages(): Observable<any[]> {
    return this.http.get<any[]>(this.apiUrl);
  }

  getHistory(): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/history`);
  }

  getMessage(id: number): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/${id}`);
  }
  downloadJson(id: number): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/${id}/export`, { responseType: 'blob' });
  }
  getProcessingStatus(id: number): Observable<any> { return this.http.get<any>(`${this.apiUrl}/${id}/processing-status`); }

  updateMessage(id: number, data: any): Observable<any> {
    return this.http.put<any>(`${this.apiUrl}/${id}`, data);
  }

  getAudit(id: number): Observable<any[]> { return this.http.get<any[]>(`${this.apiUrl}/${id}/audit`); }
  attachmentUrl(messageId: number, attachmentId: number): string { return `${this.apiUrl}/${messageId}/attachments/${attachmentId}/content`; }
  translate(text: string): Observable<{ translatedText: string }> { return this.http.post<{ translatedText: string }>(`${this.apiUrl}/translate`, { text }); }
  uploadPdf(file: File): Observable<any> { const data = new FormData(); data.append('file', file); return this.http.post<any>(`${this.apiUrl}/upload`, data); }
  loadTestData(): Observable<any> { return this.http.post<any>('/api/test-data/seed', {}); }
  getTestDataReport(): Observable<any> { return this.http.get<any>('/api/test-data/report'); }
  clearTestData(): Observable<any> { return this.http.delete<string>('/api/test-data'); }
}
