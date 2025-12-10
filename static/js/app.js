// ClaimCLAIRE Fact-Checker - Modern JavaScript

// Status checking removed - status card no longer displayed

const TRUST_BASE_WEIGHTS = {
    reliable: 1.0,
    mixed: 0.5,
    unreliable: 0.1
};

// Scroll functions
function scrollToAnalysis() {
    document.getElementById('analysisSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function scrollToUpload() {
    document.getElementById('uploadSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Analyze single claim with streaming progress
function computeEffectiveWeight(source, trustRating) {
    if (typeof source.effective_weight === 'number') {
        return source.effective_weight;
    }
    const rating = trustRating || source.trust_rating;
    const baseWeight = TRUST_BASE_WEIGHTS[rating];
    if (baseWeight === undefined) {
        return null;
    }
    const confidence = typeof source.trust_confidence === 'number' ? source.trust_confidence : 0.75;
    return baseWeight * confidence;
}

function getWeightCategory(weight) {
    if (typeof weight !== 'number' || Number.isNaN(weight)) {
        return { label: 'Weight n/a', className: 'weight-pill neutral' };
    }
    if (weight > 0.7) {
        return { label: `High • ${weight.toFixed(2)}`, className: 'weight-pill high' };
    }
    if (weight >= 0.4) {
        return { label: `Med • ${weight.toFixed(2)}`, className: 'weight-pill medium' };
    }
    return { label: `Low • ${weight.toFixed(2)}`, className: 'weight-pill low' };
}

function summarizeExplanation(explanationText) {
    if (!explanationText) return '';
    const match = explanationText.match(/[^.?!]+[.?!]/);
    return match ? match[0].trim() : explanationText.split('\n')[0];
}

function formatDomainLabel(source) {
    if (source.domain) {
        return source.domain.replace(/^https?:\/\//, '').replace(/\/$/, '');
    }
    if (source.url) {
        try {
            const parsed = new URL(source.url);
            return parsed.hostname.replace(/^www\./, '');
        } catch (e) {
            return source.url;
        }
    }
    return 'Source';
}
window.analyzeSingle = async function() {
    const claim = document.getElementById('claim_input').value.trim();
    const beforeDate = document.getElementById('before_date').value.trim();
    const progressContainer = document.getElementById('progress_container');
    const activityMessages = document.getElementById('activity_messages');
    const resultDiv = document.getElementById('single_result');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');

    if (!claim) {
        alert('Please enter a claim to verify');
        return;
    }

    // Show progress container
    progressContainer.style.display = 'block';
    resultDiv.innerHTML = '';
    activityMessages.innerHTML = '';
    analyzeBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'block';

    // Reset all stages
    document.querySelectorAll('.stage-item').forEach(stage => {
        stage.classList.remove('active', 'completed');
    });

    try {
        // Use streaming endpoint with POST body
        const response = await fetch('/analyze_single_stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({claim: claim, before_date: beforeDate || null})
        });

        if (!response.ok) {
            throw new Error(`Server error (${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const eventData = JSON.parse(line.slice(6));
                    handleProgressEvent(eventData);
                }
            }
        }
    } catch (error) {
        console.error('Error analyzing claim:', error);
        addActivityMessage('❌ Error: ' + error.message);
        resultDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    } finally {
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
};

function handleProgressEvent(event) {
    const { stage, stage_name, status, message, data, type } = event;

    // Handle completion
    if (type === 'complete') {
        // Mark both stage 4 and 5 as completed
        setStageCompleted(4);
        setStageCompleted(5);
        displaySingleResult({success: true, ...data});
        
        // Stop the spinner/loader
        const analyzeBtn = document.getElementById('analyzeBtn');
        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoader = analyzeBtn.querySelector('.btn-loader');
        analyzeBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        
        // Keep progress UI visible - don't auto-hide
        return;
    }

    // Handle error
    if (type === 'error') {
        addActivityMessage('❌ ' + message);
        return;
    }

    // Update stage badges
    if (status === 'start') {
        setStageActive(stage);
        addActivityMessage(message);
    } else if (status === 'update') {
        if (message) addActivityMessage(message);
    } else if (status === 'complete') {
        setStageCompleted(stage);
        if (message) addActivityMessage('✓ ' + message);
    }

    // Handle specific stage data
    if (stage === 1 && data?.components) {
        data.components.forEach((comp, i) => {
            addActivityMessage(`  ${i + 1}. ${comp}`);
        });
    }
}

function setStageActive(stageNum) {
    document.querySelectorAll('.stage-item').forEach(item => {
        item.classList.remove('active');
    });
    const stageItem = document.querySelector(`.stage-item[data-stage="${stageNum}"]`);
    if (stageItem) stageItem.classList.add('active');
}

function setStageCompleted(stageNum) {
    const stageItem = document.querySelector(`.stage-item[data-stage="${stageNum}"]`);
    if (stageItem) {
        stageItem.classList.remove('active');
        stageItem.classList.add('completed');
    }
}

function addActivityMessage(message) {
    const messagesDiv = document.getElementById('activity_messages');
    const messageEl = document.createElement('div');
    messageEl.className = 'activity-message';
    messageEl.textContent = message;
    messagesDiv.appendChild(messageEl);

    // Retain all messages - no limit

    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Display single result
function displaySingleResult(result) {
    const resultsDiv = document.getElementById('single_result');
    
    if (!result.success) {
        resultsDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${result.error || 'Unknown error occurred'}
            </div>
        `;
        return;
    }
    
    const verdictClass = result.verdict === 'consistent' ? 'consistent' : 'inconsistent';
    const verdictText = result.verdict === 'consistent' ? 'Consistent' : 'Inconsistent';
    
    let html = '<div class="result-card">';
    html += '<div class="result-header">';
    html += `<div class="result-claim">${escapeHtml(result.claim)}</div>`;
    html += `<span class="verdict-badge ${verdictClass}">${verdictText}</span>`;
    html += '</div>';
    
    html += '<div class="result-explanation">';
    html += '<div class="explanation-header">';
    html += '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
    html += '<strong>Explanation</strong>';
    html += '</div>';
    html += `<div class="explanation-content">${formatExplanation(escapeHtml(result.explanation))}</div>`;
    html += '</div>';
    
    // Display sources
    const sources = Array.isArray(result.sources) ? result.sources : [];
    const numSources = typeof result.num_sources === 'number' ? result.num_sources : sources.length;
    const totalSources = typeof result.total_sources_found === 'number' ? result.total_sources_found : numSources;
    const showingMore = totalSources > numSources;

    // Count referenced sources
    const referencedCount = sources.filter(s => s.referenced).length;

    html += '<div class="sources-section">';
    html += '<div class="sources-header">';
    html += '<h4>📚 Sources</h4>';
    if (showingMore) {
        html += `<span class="sources-count">Showing ${numSources} of ${totalSources} found`;
        if (referencedCount > 0) {
            html += ` (${referencedCount} cited)`;
        }
        html += '</span>';
    } else {
        html += `<span class="sources-count">${numSources} source${numSources !== 1 ? 's' : ''}`;
        if (referencedCount > 0) {
            html += ` (${referencedCount} cited)`;
        }
        html += '</span>';
    }
    html += '</div>';
    
    // Show source names summary
    if (sources.length > 0) {
        const sourceNames = sources
            .map(s => s.domain || s.title)
            .filter((name, index, self) => self.indexOf(name) === index) // unique
            .slice(0, 10); // limit to 10
        
        if (sourceNames.length > 0) {
            html += '<div class="source-names-summary">';
            html += '<strong>Sources:</strong> ';
            html += sourceNames.map(name => `<span class="source-name-tag">${escapeHtml(name)}</span>`).join(', ');
            if (sourceNames.length < sources.length) {
                html += ` <span class="source-name-more">+${sources.length - sourceNames.length} more</span>`;
            }
            html += '</div>';
        }
    }
    
    if (sources.length === 0) {
        html += '<div style="color: var(--text-secondary); font-size: 14px; padding: 16px; text-align: center;">No sources found for the current settings.</div>';
    } else {
        sources.forEach(function(source) {
            // Determine trust rating badge
            const trustRating = source.trust_rating || 'mixed';
            const trustSource = source.trust_source || 'list';
            
            let trustBadgeClass = 'badge-trust-mixed';
            let trustBadgeText = 'Mixed Reliability';
            
            if (trustRating === 'reliable') {
                trustBadgeClass = 'badge-trust-reliable';
                trustBadgeText = 'Generally Reliable';
            } else if (trustRating === 'unreliable') {
                trustBadgeClass = 'badge-trust-unreliable';
                trustBadgeText = 'Generally Unreliable';
            }
            
            // Source badge
            const sourceBadgeClass = trustSource === 'llm' ? 'badge-source-llm' : 'badge-source-list';
            const sourceBadgeText = trustSource === 'llm' ? 'LLM Classified' : 'From List';
            const sourceBadgeTitle = trustSource === 'llm' ? 'Determined by AI' : 'From known source list';
            
            // Add a class for referenced sources to highlight them
            const referencedClass = source.referenced ? ' source-referenced' : '';
            html += `<div class="source-item${referencedClass}">`;

            html += '<div class="source-header">';
            html += `<div class="source-index">[${source.index}]</div>`;
            html += '<div class="source-title-wrapper">';
            html += `<div class="source-title">${escapeHtml(source.title)}</div>`;
            if (source.domain) {
                html += `<div class="source-domain">${escapeHtml(source.domain)}</div>`;
            }
            html += '</div>';

            // Add "Cited in explanation" badge for referenced sources
            if (source.referenced) {
                html += '<div class="cited-badge" title="This source is cited in the explanation">✓ Cited</div>';
            }

            html += '</div>';
            
            const confidenceValue = typeof source.trust_confidence === 'number' ? source.trust_confidence : null;
            let effectiveWeight = typeof source.effective_weight === 'number' ? source.effective_weight : null;
            if (effectiveWeight === null && TRUST_BASE_WEIGHTS[trustRating] !== undefined) {
                const fallbackConfidence = confidenceValue !== null ? confidenceValue : 0.75;
                effectiveWeight = TRUST_BASE_WEIGHTS[trustRating] * fallbackConfidence;
            }

            html += '<div class="source-badges-row">';
            html += `<span class="badge ${trustBadgeClass}" title="Source reliability rating">${trustBadgeText}</span>`;
            html += `<span class="badge ${sourceBadgeClass}" title="${sourceBadgeTitle}">${sourceBadgeText}</span>`;
            if (confidenceValue !== null) {
                const confidencePercent = Math.round(confidenceValue * 100);
                html += `<span class="badge badge-confidence" title="Confidence: ${confidencePercent}%">${confidencePercent}% confident</span>`;
            }
            if (effectiveWeight !== null && !Number.isNaN(effectiveWeight)) {
                html += `<span class="badge badge-effective-weight" title="Effective weight = trust weight × confidence">Eff. Weight ${effectiveWeight.toFixed(2)}</span>`;
            }
            html += '</div>';
            
            // Show LLM reasoning if available
            if (trustSource === 'llm' && source.trust_reason) {
                html += '<div class="source-reasoning">';
                html += '<div class="reasoning-header">';
                html += '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7L12 12L22 7L12 2Z"></path><path d="M2 17L12 22L22 17"></path><path d="M2 12L12 17L22 12"></path></svg>';
                html += '<strong>AI Reasoning</strong>';
                html += '</div>';
                html += `<div class="reasoning-content">${escapeHtml(source.trust_reason)}</div>`;
                html += '</div>';
            }
            
            if (source.section) {
                html += `<div class="source-meta"><span class="meta-label">Section:</span> ${escapeHtml(source.section)}</div>`;
            }
            
            html += `<div class="source-preview">${escapeHtml(source.preview)}</div>`;
            
            if (source.url) {
                html += `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer" class="source-url">`;
                html += '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>';
                html += '<span>View source</span>';
                html += '</a>';
            }
            
            html += '</div>';
        });
    }
    
    html += '</div>'; // sources-section
    html += '</div>'; // result-card
    
    resultsDiv.innerHTML = html;
}

// File upload handling
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const fileInfo = document.getElementById('fileInfo');
const uploadLoading = document.getElementById('upload_loading');
const resultsDiv = document.getElementById('results');

// Click to upload
uploadArea.addEventListener('click', () => fileInput.click());
uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileInfo.style.display = 'block';
        fileInfo.innerHTML = `Selected: <strong>${escapeHtml(file.name)}</strong> (${(file.size / 1024).toFixed(2)} KB)`;
        uploadAndAnalyze(file);
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--primary-color)';
    uploadArea.style.background = 'var(--bg-secondary)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.borderColor = 'var(--border-color)';
    uploadArea.style.background = 'transparent';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = 'var(--border-color)';
    uploadArea.style.background = 'transparent';
    
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.json')) {
        fileInput.files = e.dataTransfer.files;
        fileInfo.style.display = 'block';
        fileInfo.innerHTML = `Selected: <strong>${escapeHtml(file.name)}</strong> (${(file.size / 1024).toFixed(2)} KB)`;
        uploadAndAnalyze(file);
    } else {
        alert('Please upload a JSON file');
    }
});

// Upload and analyze
async function uploadAndAnalyze(file) {
    const formData = new FormData();
    formData.append('claims_file', file);
    
    uploadLoading.style.display = 'flex';
    resultsDiv.innerHTML = '';
    uploadBtn.disabled = true;
    
    try {
        const response = await fetch('/upload_analyze', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        resultsDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    } finally {
        uploadLoading.style.display = 'none';
        uploadBtn.disabled = false;
    }
}

// Display batch results
function displayResults(data) {
    if (!data.success) {
        resultsDiv.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> ${data.error || 'Unknown error occurred'}
            </div>
        `;
        return;
    }
    
    let html = `<div class="sources-header" style="margin-bottom: 24px;">`;
    html += `<h4>Analysis Results</h4>`;
    html += `<span class="sources-count">${data.results.length} claim${data.results.length !== 1 ? 's' : ''} analyzed</span>`;
    html += `</div>`;
    
    data.results.forEach((result, i) => {
        const verdictClass = result.verdict === 'consistent' ? 'consistent' : 'inconsistent';
        const verdictText = result.verdict === 'consistent' ? 'Consistent' : 'Inconsistent';
        
        html += '<div class="result-card">';
        html += '<div class="result-header">';
        html += `<div class="result-claim">Claim ${i + 1}: ${escapeHtml(result.claim)}</div>`;
        if (result.verdict !== 'error') {
            html += `<span class="verdict-badge ${verdictClass}">${verdictText}</span>`;
        } else {
            html += `<span class="verdict-badge" style="background: #fee2e2; color: #991b1b;">Error</span>`;
        }
        html += '</div>';
        
        if (result.error) {
            html += `<div class="error-message">${escapeHtml(result.error)}</div>`;
        } else if (result.explanation) {
            html += `<div class="result-explanation">${escapeHtml(result.explanation)}</div>`;
        }
        
        html += '</div>';
    });
    
    resultsDiv.innerHTML = html;
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Format explanation text for better readability
function formatExplanation(text) {
    // Split by sentences and add line breaks for better readability
    // Also highlight source references like [1, 2] or [5]
    let formatted = text
        .replace(/(\[[\d,\s]+\])/g, '<span class="source-ref">$1</span>')
        .replace(/\.\s+/g, '.<br><br>')
        .replace(/\n\n+/g, '<br><br>');
    
    return formatted;
}

