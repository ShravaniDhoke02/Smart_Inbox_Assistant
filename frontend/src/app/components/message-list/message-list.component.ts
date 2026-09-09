import { Component, OnInit } from '@angular/core';
import { MessageService } from '../../services/message.service';

@Component({
  selector: 'app-message-list',
  template: `
    <h2>Pending Review Inbox</h2>
    <table mat-table [dataSource]="messages" class="mat-elevation-z8">
      
      <ng-container matColumnDef="subject">
        <th mat-header-cell *matHeaderCellDef> Subject </th>
        <td mat-cell *matCellDef="let msg"> {{msg.subject}} </td>
      </ng-container>

      <ng-container matColumnDef="category">
        <th mat-header-cell *matHeaderCellDef> Category </th>
        <td mat-cell *matCellDef="let msg"> 
          <mat-chip-listbox>
            <mat-chip>{{msg.category}}</mat-chip>
          </mat-chip-listbox>
        </td>
      </ng-container>

      <ng-container matColumnDef="confidence">
        <th mat-header-cell *matHeaderCellDef> Confidence </th>
        <td mat-cell *matCellDef="let msg"> {{msg.confidenceScore | percent}} </td>
      </ng-container>

      <ng-container matColumnDef="actions">
        <th mat-header-cell *matHeaderCellDef> Actions </th>
        <td mat-cell *matCellDef="let msg">
          <button mat-raised-button *ngIf="msg.category !== 'Not Relevant'">Review</button>
          <mat-card *ngIf="msg.category === 'Not Relevant'" class="not-relevant-card">
            <mat-card-content>
              <p>This data is not relevant to the application. No further action is required.</p>
            </mat-card-content>
          </mat-card>
        </td>
      </ng-container>

      <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
      <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
    </table>
  `,
  styles: [`
    table { width: 100%; }
    .not-relevant-card { margin: 8px; background-color: #f5f5f5; }
  `]
})
export class MessageListComponent implements OnInit {
  messages: any[] = [];
  displayedColumns: string[] = ['subject', 'category', 'confidence', 'actions'];

  constructor(private messageService: MessageService) {}

  ngOnInit() {
    this.messageService.getPendingMessages().subscribe(
      data => this.messages = data,
      err => console.error(err)
    );
  }
}
