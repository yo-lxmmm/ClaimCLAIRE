// Enhanced Timeline Analysis - Frontend Logic

// Set default dates on page load
document.addEventListener('DOMContentLoaded', function() {
    const startDateInput = document.getElementById('start_date');
    const endDateInput = document.getElementById('end_date');

    const today = new Date();
    const fiveYearsAgo = new Date(today.getFullYear() - 5, 0, 1); // Jan 1, 5 years ago

    // Set defaults
    startDateInput.value = formatDateForInput(fiveYearsAgo);
    endDateInput.value = formatDateForInput(today);

    // Set min/max constraints
    startDateInput.setAttribute('min', '1900-01-01');
    startDateInput.setAttribute('max', formatDateForInput(today));
    endDateInput.setAttribute('min', '1900-01-01');
    endDateInput.setAttribute('max', formatDateForInput(today));
});

function formatDateForInput(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function addTimelineActivityMessage(message) {
    const messagesDiv = document.getElementById('timeline_activity_messages');
    const messageEl = document.createElement('div');
    messageEl.className = 'activity-message';
    messageEl.textContent = message;
    messagesDiv.appendChild(messageEl);

    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function clearTimelineActivityMessages() {
    const messagesDiv = document.getElementById('timeline_activity_messages');
    messagesDiv.innerHTML = '';
}

async function analyzeTimeline() {
    const claim = document.getElementById('timeline_claim').value.trim();
    const startDateStr = document.getElementById('start_date').value;
    const endDateStr = document.getElementById('end_date').value;

    if (!claim) {
        alert('Please enter a claim to analyze');
        return;
    }

    if (!startDateStr || !endDateStr) {
        alert('Please select both start and end dates');
        return;
    }

    const startDate = new Date(startDateStr);
    const endDate = new Date(endDateStr);

    if (startDate >= endDate) {
        alert('Start date must be before end date');
        return;
    }

    // Show loading
    const loadingDiv = document.getElementById('timeline_loading');
    const resultDiv = document.getElementById('timeline_result');
    const analyzeBtn = document.getElementById('analyzeTimelineBtn');

    loadingDiv.style.display = 'block';
    resultDiv.innerHTML = '';
    analyzeBtn.disabled = true;
    clearTimelineActivityMessages();

    try {
        console.log('Starting enhanced timeline analysis...', {claim, startDate: startDateStr, endDate: endDateStr});

        // Add initial activity messages
        addTimelineActivityMessage('Starting timeline analysis...');
        addTimelineActivityMessage('Executing targeted web searches for timeline events...');

        // Simulate progress updates while waiting
        const progressInterval = setInterval(() => {
            const messages = [
                'Analyzing search results...',
                'Extracting timeline events with AI...',
                'Filtering and prioritizing key events...',
                'Verifying claim at event dates...',
                'Generating contextual explanations...',
                'Synthesizing narrative summary...'
            ];
            const randomMsg = messages[Math.floor(Math.random() * messages.length)];
            addTimelineActivityMessage(randomMsg);
        }, 8000); // Add a message every 8 seconds

        const response = await fetch('/analyze_timeline', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                claim: claim,
                start_date: startDateStr,
                end_date: endDateStr
            })
        });

        clearInterval(progressInterval);

        const result = await response.json();
        console.log('Timeline result:', result);

        if (result.success) {
            addTimelineActivityMessage(`Timeline complete! Analyzed ${result.metadata?.events_verified || 0} events.`);
            displayEnhancedTimeline(result);
        } else {
            showError(result.error || 'Unknown error occurred');
        }
    } catch (error) {
        console.error('Timeline analysis error:', error);
        showError(error.message);
    } finally {
        loadingDiv.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

function displayEnhancedTimeline(data) {
    const resultsDiv = document.getElementById('timeline_result');

    let html = '<div class="timeline-results">';

    // Header
    html += '<div class="timeline-header">';
    html += `<h2>${escapeHtml(data.claim)}</h2>`;
    html += `<div class="timeline-range">${data.time_range.start} → ${data.time_range.end}</div>`;

    // Category badge
    const categoryLabel = formatCategoryLabel(data.category);
    html += `<div class="category-badge ${data.category}">${categoryLabel}</div>`;
    html += '</div>';

    // Key Insight
    if (data.key_insight) {
        html += '<div class="key-insight">';
        html += '<strong>💡 Key Insight</strong>';
        html += `<p>${escapeHtml(data.key_insight)}</p>`;
        html += '</div>';
    }

    // Overall Summary
    if (data.summary) {
        html += '<div class="narrative-summary">';
        html += '<h3>📖 Timeline Summary</h3>';
        html += `<p>${escapeHtml(data.summary)}</p>`;
        html += '</div>';
    }

    // Current Status
    if (data.current_status) {
        html += '<div class="current-status">';
        html += '<strong>Current Status</strong>';
        html += `<p>${escapeHtml(data.current_status)}</p>`;
        html += '</div>';
    }

    // Events timeline
    if (data.events && data.events.length > 0) {
        html += '<div class="timeline-container">';
        html += '<div class="timeline-line"></div>';

        data.events.forEach((event, index) => {
            html += renderTimelineEvent(event, index);
        });

        html += '</div>'; // timeline-container
    } else {
        html += '<div class="no-transitions">';
        html += '<p><strong>No significant events identified</strong></p>';
        html += '<p>Insufficient data available to construct a timeline for this claim.</p>';
        html += '</div>';
    }

    // Metadata
    if (data.metadata) {
        html += '<div class="timeline-summary">';
        html += `<div class="summary-stat">📊 ${data.metadata.events_verified || 0} Events Analyzed</div>`;
        html += `<div class="summary-stat">🔍 ${data.metadata.total_searches || 0} Web Searches</div>`;
        html += `<div class="summary-stat">⏱️ ${(data.metadata.duration_seconds || 0).toFixed(1)}s</div>`;
        html += '</div>';
    }

    html += '</div>'; // timeline-results

    resultsDiv.innerHTML = html;
}

function renderTimelineEvent(event, index) {
    let html = '<div class="timeline-event">';

    // Event marker (dot on timeline)
    html += `<div class="event-marker ${event.verdict}"></div>`;

    // Event type badge
    const eventTypeLabel = formatEventTypeLabel(event.event_type);
    html += `<div class="event-type-badge">${eventTypeLabel}</div>`;

    // Event card
    html += '<div class="event-card">';

    // Header with date and verdict
    html += '<div class="event-header">';
    html += `<div class="event-date">📅 ${formatDate(event.date)}</div>`;
    html += `<div class="event-verdict-badge ${event.verdict}">${event.verdict.toUpperCase()}</div>`;
    html += '</div>';

    // Event description
    html += `<div class="event-description">${escapeHtml(event.event_description)}</div>`;

    // Event explanation
    if (event.explanation) {
        html += `<div class="event-explanation">${escapeHtml(event.explanation)}</div>`;
    }

    // Event metadata
    html += '<div class="event-meta">';
    html += `<div class="meta-item">`;
    html += `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>`;
    html += `<span class="meta-value">${event.num_sources}</span> sources`;
    html += `</div>`;

    html += `<div class="meta-item">`;
    html += `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`;
    html += `<span class="meta-value">${event.reliable_count}</span> reliable`;
    html += `</div>`;

    if (event.mixed_count > 0) {
        html += `<div class="meta-item">`;
        html += `<span class="meta-value">${event.mixed_count}</span> mixed`;
        html += `</div>`;
    }

    if (event.unreliable_count > 0) {
        html += `<div class="meta-item">`;
        html += `<span class="meta-value">${event.unreliable_count}</span> unreliable`;
        html += `</div>`;
    }

    html += '</div>'; // event-meta

    // Event sources (collapsible)
    if (event.key_sources && event.key_sources.length > 0) {
        html += '<details class="event-sources">';
        html += `<summary>View ${event.key_sources.length} Key Sources</summary>`;
        html += '<ul class="source-list">';

        event.key_sources.forEach(source => {
            html += `<li class="source-item ${source.trust_rating}">`;
            html += `<div class="source-title">${escapeHtml(source.title)}</div>`;
            if (source.url) {
                html += `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener" class="source-url">${escapeHtml(source.url)}</a>`;
            }
            html += `</li>`;
        });

        html += '</ul>';
        html += '</details>';
    }

    html += '</div>'; // event-card
    html += '</div>'; // timeline-event

    return html;
}

function formatCategoryLabel(category) {
    const labels = {
        'evolving_fact': '📊 Evolving Fact',
        'debunking_journey': '🔍 Debunking Journey',
        'stable_truth': '✅ Stable Truth',
        'stable_falsehood': '❌ Stable Falsehood',
        'narrative_mutation': '🔄 Narrative Mutation',
        'insufficient_data': '⚠️ Insufficient Data'
    };
    return labels[category] || category;
}

function formatEventTypeLabel(eventType) {
    const labels = {
        'claim_origin': '🎯 Origin',
        'fact_check': '🔍 Fact Check',
        'new_evidence': '📰 New Evidence',
        'official_update': '📢 Official Update',
        'retraction': '↩️ Retraction'
    };
    return labels[eventType] || eventType;
}

function formatDate(dateStr) {
    // Add 1 day to compensate for the backend using date-1 for temporal search
    const date = new Date(dateStr);
    date.setDate(date.getDate() + 1);
    const options = {year: 'numeric', month: 'long', day: 'numeric'};
    return date.toLocaleDateString('en-US', options);
}

function showError(message) {
    const resultsDiv = document.getElementById('timeline_result');
    resultsDiv.innerHTML = `
        <div class="error-message">
            <strong>Error:</strong> ${escapeHtml(message)}
        </div>
    `;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
