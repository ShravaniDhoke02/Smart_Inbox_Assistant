let currentMessage = null;
let currentMessageId = null;
let savedMessage = null;

document.addEventListener('DOMContentLoaded', () => {
    fetchMessages();
});

async function fetchMessages() {
    try {
        const [queueResponse, historyResponse] = await Promise.all([
            fetch('/api/messages'),
            fetch('/api/messages/history')
        ]);
        renderQueue(await queueResponse.json());
        await historyResponse.json();
    } catch (error) {
        console.error('Error fetching messages:', error);
    }
}

async function showHistory() {
    try {
        const response = await fetch('/api/messages/history');
        if (!response.ok) throw new Error(`History request failed: ${response.status}`);
        renderHistory(await response.json());
        document.getElementById('queueView').classList.add('hidden');
        document.getElementById('reviewView').classList.add('hidden');
        document.getElementById('approvedView').classList.add('hidden');
        document.getElementById('historyView').classList.remove('hidden');
        document.getElementById('headerTitle').textContent = 'History';
        document.getElementById('queueNav').className = 'flex items-center px-6 py-3 hover:text-white hover:bg-slate-800 transition-colors';
        document.getElementById('historyNav').className = 'flex items-center px-6 py-3 text-indigo-400 bg-slate-800 border-r-4 border-indigo-500';
    } catch (error) {
        console.error('Error fetching history:', error);
    }
}

async function showApproved() {
    try {
        const response = await fetch('/api/messages/approved');
        if (!response.ok) throw new Error(`Approved request failed: ${response.status}`);
        renderApproved(await response.json());
        document.getElementById('queueView').classList.add('hidden');
        document.getElementById('reviewView').classList.add('hidden');
        document.getElementById('historyView').classList.add('hidden');
        document.getElementById('approvedView').classList.remove('hidden');
        document.getElementById('headerTitle').textContent = 'Approved & Saved';
    } catch (error) {
        console.error('Error fetching approved records:', error);
    }
}

function renderHistory(messages) {
    const tbody = document.getElementById('historyTableBody');
    const emptyState = document.getElementById('historyEmptyState');
    tbody.innerHTML = '';
    emptyState.classList.toggle('hidden', messages.length > 0);
    messages.forEach((msg) => {
        const attachment = (msg.attachments || [])[0];
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-50 transition-colors cursor-pointer';
        tr.onclick = () => openMessage(msg.id, true);
        tr.innerHTML = `<td class="px-6 py-4 text-sm text-slate-500">${new Date(msg.receivedDate).toLocaleString()}</td>
            <td class="px-6 py-4 text-sm font-medium text-slate-900">${attachment ? attachment.filename : (msg.subject || 'No file')}</td>
            <td class="px-6 py-4 text-sm text-slate-700">${msg.category || 'Unclassified'}</td>
            <td class="px-6 py-4 text-sm font-bold text-emerald-600">${msg.status}</td>
            <td class="px-6 py-4 text-right text-sm font-medium text-indigo-600">View</td>`;
        tbody.appendChild(tr);
    });
}

function renderApproved(messages) {
    const tbody = document.getElementById('approvedTableBody');
    const emptyState = document.getElementById('approvedEmptyState');
    tbody.innerHTML = '';
    emptyState.classList.toggle('hidden', messages.length > 0);
    messages.forEach((msg) => {
        const attachment = (msg.attachments || [])[0];
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-50 transition-colors cursor-pointer';
        tr.onclick = () => openMessage(msg.id, true);
        tr.innerHTML = `<td class="px-6 py-4 text-sm text-slate-500">${new Date(msg.receivedDate).toLocaleString()}</td>
            <td class="px-6 py-4 text-sm font-medium text-slate-900">${attachment ? attachment.filename : (msg.subject || 'No file')}</td>
            <td class="px-6 py-4 text-sm text-slate-700">${msg.category || 'Unclassified'}</td>
            <td class="px-6 py-4 text-sm font-bold text-emerald-600">${msg.confidenceScore ? Math.round(msg.confidenceScore * 100) + '%' : '-'}</td>
            <td class="px-6 py-4 text-right text-sm font-medium text-indigo-600">View</td>`;
        tbody.appendChild(tr);
    });
}

async function clearTestData() {
    if (!confirm('Remove only synthetic demo messages from the review queue?')) return;
    try {
        const res = await fetch('/api/test-data', { method: 'DELETE' });
        alert(await res.text());
        fetchMessages();
    } catch (e) {
        console.error(e);
        alert('Unable to clear the synthetic batch.');
    }
}

async function seedDemoData() {
    try {
        const response = await fetch('/api/test-data/seed', { method: 'POST' });
        if (!response.ok) throw new Error(await response.text());
        const result = await response.json();
        alert(`Queued ${result.created} synthetic documents. Refreshing while AI processing completes.`);
        await refreshUntilProcessed();
    } catch (error) {
        console.error(error);
        alert(error.message || 'Unable to load the synthetic demo corpus.');
    }
}

function parseStoredJson(value, fallback) {
    if (!value || typeof value !== 'string') return value || fallback;
    try { return JSON.parse(value); } catch (_) { return fallback; }
}

async function uploadPdfFile(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    const uploadButton = document.getElementById('uploadPdfButton');
    const originalButtonText = uploadButton ? uploadButton.textContent : '';
    if (uploadButton) {
        uploadButton.disabled = true;
        uploadButton.textContent = 'Uploading...';
        uploadButton.classList.add('opacity-60', 'cursor-wait');
    }

    setUploadProgress(0, `Preparing ${file.name}…`);

    try {
        await sendUploadWithProgress(formData, file.name);

        setUploadProgress(100, 'Upload complete. Adding the document to the review queue…');
        await refreshUntilProcessed();
        setUploadProgress(100, `${file.name} is ready for review.`);
    } catch (error) {
        console.error('Upload failed:', error);
        setUploadProgress(0, `Upload failed: ${error.message || 'Unable to upload the PDF.'}`);
        alert(error.message || 'Unable to upload the PDF.');
    } finally {
        if (uploadButton) {
            uploadButton.disabled = false;
            uploadButton.textContent = originalButtonText;
            uploadButton.classList.remove('opacity-60', 'cursor-wait');
        }
    }
}

function setUploadProgress(percent, status) {
    const panel = document.getElementById('uploadProgressPanel');
    const bar = document.getElementById('uploadProgressBar');
    const label = document.getElementById('uploadProgressPercent');
    const statusText = document.getElementById('uploadProgressStatus');
    if (!panel || !bar || !label || !statusText) return;

    const safePercent = Math.max(0, Math.min(100, Math.round(percent)));
    panel.classList.remove('hidden');
    bar.style.width = `${safePercent}%`;
    label.textContent = `${safePercent}%`;
    statusText.textContent = status;
}

function sendUploadWithProgress(formData, filename) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/messages/upload');

        xhr.upload.addEventListener('progress', (event) => {
            if (!event.lengthComputable) {
                setUploadProgress(0, `Uploading ${filename}…`);
                return;
            }
            const percent = (event.loaded / event.total) * 100;
            setUploadProgress(percent, `Uploading ${filename}…`);
        });

        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve();
            } else {
                reject(new Error(xhr.responseText || 'Upload failed'));
            }
        });
        xhr.addEventListener('error', () => reject(new Error('Network error while uploading the PDF.')));
        xhr.addEventListener('abort', () => reject(new Error('Upload was cancelled.')));
        xhr.send(formData);
    });
}

async function refreshUntilProcessed() {
    for (let attempt = 0; attempt < 30; attempt += 1) {
        await fetchMessages();
        const response = await fetch('/api/messages');
        if (response.ok && (await response.json()).length > 0) return;
        await new Promise((resolve) => setTimeout(resolve, 1000));
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('pdfUploadInput');
    if (uploadInput) {
        uploadInput.addEventListener('change', (event) => {
            const file = event.target.files && event.target.files[0];
            if (file) {
                uploadPdfFile(file);
                event.target.value = '';
            }
        });
    }
    fetchMessages();
});

function renderQueue(messages) {
    const tbody = document.getElementById('queueTableBody');
    const emptyState = document.getElementById('emptyState');
    tbody.innerHTML = '';

    if (!messages || messages.length === 0) {
        emptyState.classList.remove('hidden');
        return;
    }
    
    emptyState.classList.add('hidden');

    messages.forEach((msg, index) => {
        const date = new Date(msg.receivedDate).toLocaleDateString() + ' ' + new Date(msg.receivedDate).toLocaleTimeString();
        const confColor = msg.confidenceScore > 0.8 ? 'text-emerald-600' : 'text-amber-600';
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-50 transition-colors cursor-pointer animate-fade-in';
        tr.style.animationDelay = `${index * 0.05}s`;
        tr.onclick = () => openMessage(msg.id);
        
        tr.innerHTML = `
            <td class="px-6 py-4 whitespace-nowrap text-sm text-slate-500">${date}</td>
            <td class="px-6 py-4 text-sm font-medium text-slate-900 truncate max-w-xs">${msg.subject || 'No Subject'}</td>
            <td class="px-6 py-4 text-sm text-slate-700">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-100 text-indigo-800">
                    ${msg.category || 'Unclassified'}
                </span>
            </td>
            <td class="px-6 py-4 text-sm font-bold ${confColor}">${msg.confidenceScore ? Math.round(msg.confidenceScore * 100) + '%' : '-'}</td>
            <td class="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <button class="text-indigo-600 hover:text-indigo-900 bg-indigo-50 px-3 py-1 rounded-md transition-colors hover:bg-indigo-100">Review</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

async function openMessage(id, readOnly = false) {
    try {
        const response = await fetch(`/api/messages/${id}`);
        const msg = await response.json();
        currentMessage = msg;
        currentMessageId = id;
        
        // Hide queue, show review
        document.getElementById('queueView').classList.add('hidden');
        document.getElementById('historyView').classList.add('hidden');
        document.getElementById('approvedView').classList.add('hidden');
        document.getElementById('reviewView').classList.remove('hidden');
        document.getElementById('approveButton').classList.toggle('hidden', readOnly);
        document.getElementById('headerTitle').textContent = `Review: ${msg.subject}`;
        
        setDocumentViewer(msg);
        
        // Populate AI Form
        document.getElementById('editCategory').value = msg.category || '';
        document.getElementById('lblConfidence').textContent = msg.confidenceScore ? Math.round(msg.confidenceScore * 100) + '%' : '-';
        document.getElementById('lblReason').textContent = msg.classificationReason || 'N/A';
        document.getElementById('editSummary').value = msg.aiSummary || '';

        const sourceTrace = parseStoredJson(msg.sourceTrace, {});
        const literature = parseStoredJson(msg.literatureScreening, {});
        const reviewMeta = document.getElementById('reviewMeta');
        reviewMeta.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <div class="text-[10px] uppercase tracking-wide text-slate-500 mb-1">Language</div>
                    <div class="font-semibold text-slate-800">${msg.language || 'Unknown'}</div>
                </div>
                <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <div class="text-[10px] uppercase tracking-wide text-slate-500 mb-1">PDF Type</div>
                    <div class="font-semibold text-slate-800">${Array.isArray(msg.pdfTypes) ? msg.pdfTypes.join(', ') : (msg.pdfTypes || 'N/A')}</div>
                </div>
                <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <div class="text-[10px] uppercase tracking-wide text-slate-500 mb-1">Traceability</div>
                    <div class="font-semibold text-slate-800">${sourceTrace.provenanceRule || 'Not available'}</div>
                </div>
            </div>
        `;

        const literatureSection = document.getElementById('literatureSection');
        literatureSection.innerHTML = `
            <div class="bg-amber-50 border border-amber-200 rounded-xl p-4">
                <h4 class="text-xs font-bold uppercase tracking-wider text-amber-700 mb-2">Literature screening</h4>
                <div class="text-sm text-amber-900 font-medium">${literature.screeningStatus || 'Not applicable'}</div>
                <div class="text-sm text-amber-800 mt-2">${literature.caseSummary || literature.recommendation || 'No literature-like content detected.'}</div>
            </div>
        `;

        const tableSection = document.getElementById('tableSection');
        let tableData = msg.tables;
        if (typeof tableData === 'string') {
            try {
                tableData = JSON.parse(tableData);
            } catch (error) {
                tableData = [];
            }
        }
        if (Array.isArray(tableData) && tableData.length > 0 && tableData[0] && typeof tableData[0] !== 'string') {
            const tableHtml = tableData.map((table) => {
                const rows = table.rows || [];
                const firstRows = rows.slice(0, 5).map((row) => `<tr>${row.map((cell) => `<td class="border border-slate-200 px-2 py-1 text-left text-xs">${cell}</td>`).join('')}</tr>`).join('');
                return `
                    <div class="bg-slate-50 border border-slate-200 rounded-xl p-4 mb-2">
                        <h5 class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Table: ${table.attachment || 'Attachment'}</h5>
                        <table class="w-full border-collapse border border-slate-200 text-xs">
                            <tbody>${firstRows}</tbody>
                        </table>
                    </div>
                `;
            }).join('');
            tableSection.innerHTML = `<div class="bg-white border border-slate-200 rounded-xl p-4"><h4 class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Structured tables</h4>${tableHtml}</div>`;
        } else {
            tableSection.innerHTML = `<div class="bg-white border border-slate-200 rounded-xl p-4"><h4 class="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Structured tables</h4><div class="text-sm text-slate-600">${Array.isArray(tableData) ? tableData.join(', ') : 'No table extracted.'}</div></div>`;
        }

        const imageSection = document.getElementById('imageSection');
        const imageNotes = Array.isArray(msg.imageNotes) ? msg.imageNotes : (msg.imageNotes ? [msg.imageNotes] : []);
        imageSection.innerHTML = `
            <div class="bg-rose-50 border border-rose-200 rounded-xl p-4">
                <h4 class="text-xs font-bold uppercase tracking-wider text-rose-700 mb-2">Image and OCR review</h4>
                <ul class="list-disc list-inside text-sm text-rose-900 space-y-1">
                    ${(Array.isArray(imageNotes) ? imageNotes : ['No image review required.']).map((note) => `<li>${note}</li>`).join('')}
                </ul>
            </div>
        `;

        const auditSection = document.getElementById('auditSection');
        const auditResponse = await fetch(`/api/messages/${id}/audit`);
        const auditEvents = auditResponse.ok ? await auditResponse.json() : [];
        auditSection.innerHTML = `<div class="bg-slate-50 border border-slate-200 rounded-xl p-4"><h4 class="text-xs font-bold uppercase tracking-wider text-slate-600 mb-2">Audit trail</h4>${auditEvents.length ? `<ul class="space-y-2 text-sm text-slate-700">${auditEvents.map((event) => `<li><strong>${event.action}</strong> - ${event.performedBy} - ${new Date(event.timestamp).toLocaleString()}<br><span class="text-xs text-slate-500">${event.details || ''}</span></li>`).join('')}</ul>` : '<p class="text-sm text-slate-500">No audit events recorded.</p>'}</div>`;
        
        // Populate Facts
        const factsContainer = document.getElementById('factsContainer');
        factsContainer.innerHTML = '';
        
        const factsByKey = new Map();
        const uniqueFacts = Array.from(new Map((msg.extractedFacts || []).map((fact) => {
            const key = `${fact.factGroup || ''}\u0000${fact.fieldName || ''}`.toLowerCase();
            const existing = factsByKey.get(key);
            const isFallback = (value) => String(value || '').toLowerCase().includes('not stated');
            const selected = !existing || (isFallback(existing.fieldValue) && !isFallback(fact.fieldValue)) ? fact : existing;
            factsByKey.set(key, selected);
            return [key, selected];
        })).values());

        if (uniqueFacts.length > 0) {
            // Group by factGroup
            const grouped = uniqueFacts.reduce((acc, fact) => {
                if (!acc[fact.factGroup]) acc[fact.factGroup] = [];
                acc[fact.factGroup].push(fact);
                return acc;
            }, {});
            
            for (const [group, facts] of Object.entries(grouped)) {
                let groupHtml = `<div class="mb-4"><h4 class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 ml-1">${group}</h4><div class="space-y-3">`;
                facts.forEach((fact, idx) => {
                    const confClass = fact.confidence > 0.8 ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800';
                    groupHtml += `
                    <div class="bg-white border border-slate-200 rounded-lg p-4 shadow-sm hover:border-indigo-300 transition-colors group">
                        <div class="flex justify-between items-start mb-2">
                            <label class="text-xs font-medium text-slate-700">${fact.fieldName}</label>
                            <span class="text-[10px] px-2 py-0.5 rounded-full ${confClass} font-medium">${Math.round((fact.confidence || 0) * 100)}%</span>
                        </div>
                        <input type="text" id="fact_${fact.id || idx}" class="w-full text-sm text-slate-900 border-0 border-b border-slate-200 focus:ring-0 focus:border-indigo-500 bg-transparent p-0 pb-1 font-medium mb-2" value="${fact.fieldValue || ''}">
                        <div class="text-[10px] text-slate-400 group-hover:text-indigo-500 transition-colors flex items-center">
                            <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            Source: ${fact.sourceReference || 'N/A'}
                        </div>
                    </div>`;
                });
                groupHtml += `</div></div>`;
                factsContainer.innerHTML += groupHtml;
            }
        } else {
            factsContainer.innerHTML = '<p class="text-sm text-slate-500 italic">No facts extracted for this category.</p>';
        }
        
    } catch (error) {
        console.error('Error opening message:', error);
    }
}

function setDocumentViewer(message) {
    const section = document.getElementById('docPdfViewerSection');
    const unavailable = document.getElementById('docPdfUnavailable');
    const viewer = document.getElementById('docPdfViewer');
    const openLink = document.getElementById('docPdfOpenLink');
    if (!section || !unavailable || !viewer || !openLink) return;

    const attachment = (message.attachments || []).find((item) =>
        item && item.id && (
            item.fileType === 'application/pdf' ||
            (item.filename || '').toLowerCase().endsWith('.pdf')
        )
    );

    if (!attachment || !message.id) {
        section.classList.add('hidden');
        unavailable.classList.remove('hidden');
        unavailable.classList.add('grid');
        viewer.src = 'about:blank';
        openLink.href = '#';
        return;
    }

    const sourceUrl = `/api/messages/${encodeURIComponent(message.id)}/attachments/${encodeURIComponent(attachment.id)}/content`;
    viewer.src = sourceUrl;
    openLink.href = sourceUrl;
    openLink.textContent = `Open ${attachment.filename || 'PDF'} in new tab`;
    unavailable.classList.add('hidden');
    unavailable.classList.remove('grid');
    section.classList.remove('hidden');
}

async function translateCurrentRecord() {
    const button = document.getElementById('translateButton');
    const output = document.getElementById('translationText');
    if (!currentMessage || !output) return;
    button.disabled = true;
    button.textContent = 'Translating...';
    try {
        const response = await fetch('/api/messages/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: currentMessage.translatedText || currentMessage.body || '' })
        });
        if (!response.ok) throw new Error(`Translation failed: ${response.status}`);
        const result = await response.json();
        output.textContent = result.translatedText || 'Translation unavailable.';
        button.textContent = 'Translated';
    } catch (error) {
        console.error('Translation failed:', error);
        button.disabled = false;
        button.textContent = 'Translate to English';
        output.textContent = 'Translation unavailable. Please review the original text.';
    }
}

function showQueue() {
    document.getElementById('reviewView').classList.add('hidden');
    document.getElementById('historyView').classList.add('hidden');
    document.getElementById('approvedView').classList.add('hidden');
    document.getElementById('queueView').classList.remove('hidden');
    document.getElementById('headerTitle').textContent = 'Review Queue';
    document.getElementById('queueNav').className = 'flex items-center px-6 py-3 text-indigo-400 bg-slate-800 border-r-4 border-indigo-500';
    document.getElementById('historyNav').className = 'flex items-center px-6 py-3 hover:text-white hover:bg-slate-800 transition-colors';
    currentMessage = null;
    currentMessageId = null;
    fetchMessages();
}

async function approveRecord() {
    if (!currentMessage) return;
    
    // Gather updated values
    currentMessage.category = document.getElementById('editCategory').value;
    currentMessage.aiSummary = document.getElementById('editSummary').value;
    
    // Gather facts
    if (currentMessage.extractedFacts) {
        currentMessage.extractedFacts.forEach((fact, idx) => {
            const el = document.getElementById(`fact_${fact.id || idx}`);
            if (el) fact.fieldValue = el.value;
        });
    }
    
    try {
        const response = await fetch(`/api/messages/${currentMessageId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentMessage)
        });
        
        if (response.ok) {
            savedMessage = await response.json();
            alert('Record approved and saved successfully!');
            document.getElementById('approveButton').classList.add('hidden');
            document.getElementById('savedDataButton').classList.remove('hidden');
            renderSavedRecord(savedMessage);
        } else {
            const errorText = await response.text();
            alert(`Failed to save record: ${errorText || response.statusText}`);
        }
    } catch (error) {
        console.error('Error saving record:', error);
        alert('Error saving record.');
    }
}

function renderSavedRecord(record) {
    const facts = (record.extractedFacts || []).map((fact) =>
        `${fact.factGroup}.${fact.fieldName}: ${fact.fieldValue || 'Not stated'}`
    );
    document.getElementById('savedRecordContent').innerHTML = `
        <div><strong>Record ID:</strong> ${record.id}</div>
        <div><strong>Status:</strong> ${record.status}</div>
        <div><strong>Subject:</strong> ${record.subject || 'No Subject'}</div>
        <div><strong>Classification:</strong> ${record.category || 'Unclassified'}</div>
        <div><strong>Confidence:</strong> ${record.confidenceScore ? Math.round(record.confidenceScore * 100) + '%' : '-'}</div>
        <div><strong>Uploaded file:</strong> ${(record.attachments || []).map((attachment) => attachment.filename).join(', ') || 'None'}</div>
        <div><strong>Extracted facts:</strong> ${facts.length ? facts.join(' | ') : 'None'}</div>
    `;
    document.getElementById('savedRecordPanel').classList.remove('hidden');
}

function showSavedRecord() {
    if (savedMessage) renderSavedRecord(savedMessage);
    document.getElementById('savedRecordPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
