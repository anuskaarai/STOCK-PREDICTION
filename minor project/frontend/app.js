// API Configuration
let API_BASE_URL = 'http://127.0.0.1:8000';
const FORECAST_HOURS = 24;

// DOM Elements
const tickerInput = document.getElementById('tickerInput');
const predictBtn = document.getElementById('predictBtn');
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loaderText');
const displayTicker = document.getElementById('displayTicker');
const displayPeriod = document.getElementById('displayPeriod');
const forecastBody = document.getElementById('forecastBody');
const quickStats = document.getElementById('quickStats');
const exportCsvBtn = document.getElementById('exportCsvBtn');

const valLoss = document.getElementById('valLoss');
const nextClosePrice = document.getElementById('nextClosePrice');
const nextDirection = document.getElementById('nextDirection');

const chipNextClose = document.getElementById('chipNextClose');

// Modal Elements
const navHistoryBtn = document.getElementById('navHistoryBtn');
const historyModal = document.getElementById('historyModal');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const historyList = document.getElementById('historyList');

// Chart Instance
let predictionChart = null;
let lastPredictions = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    tickerInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handlePrediction();
    });

    predictBtn.addEventListener('click', handlePrediction);
    exportCsvBtn.addEventListener('click', exportCSV);

    // History logic
    navHistoryBtn.addEventListener('click', (e) => {
        e.preventDefault();
        renderHistory();
        historyModal.classList.remove('hidden');
    });

    closeHistoryBtn.addEventListener('click', () => {
        historyModal.classList.add('hidden');
    });

    // Close modals on outside click
    window.addEventListener('click', (e) => {
        if (e.target === historyModal) historyModal.classList.add('hidden');
    });
});

/**
 * Main prediction handler
 */
async function handlePrediction() {
    let ticker = tickerInput.value.trim().toUpperCase();
    if (!ticker) {
        showToast('Please enter a valid stock ticker.', 'error');
        return;
    }
    
    // Auto-append .NS for Indian stocks if no suffix is provided
    if (!ticker.includes('.')) {
        ticker = ticker + '.NS';
    }

    showLoader(true, 'Connecting to Quant-Edge Engine...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker, forecast_hours: FORECAST_HOURS })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Prediction failed');
        }

        const data = await response.json();
        updateUI(data);
        // Save to History
        saveToHistory(ticker, data);
        showToast(`${ticker} forecast generated successfully!`, 'success');
    } catch (error) {
        console.error('Error:', error);
        showToast(`${error.message}`, 'error');
    } finally {
        showLoader(false);
    }
}

/**
 * Update all UI components with fetched data
 */
function updateUI(data) {
    const { ticker, predictions, metrics, historical } = data;
    lastPredictions = { ticker, predictions };

    // Update Headers
    displayTicker.textContent = `${ticker} Predictions`;
    displayPeriod.textContent = `Next ${predictions.length} Trading Hours`;

    // Update Metric Cards
    valLoss.textContent = metrics.val_loss.toFixed(4);

    // Update Next Hour Card
    const nextHour = predictions[0];
    nextClosePrice.textContent = nextHour.pred_close.toFixed(2);
    
    nextDirection.className = `direction ${nextHour.close_direction.toLowerCase()}`;
    nextDirection.innerHTML = `<i class="fas fa-arrow-${nextHour.close_direction === 'UP' ? 'up' : 'down'}"></i> ${nextHour.close_direction}`;

    // Update Quick Stats Row
    quickStats.style.display = 'grid';
    chipNextClose.textContent = '₹' + nextHour.pred_close.toFixed(2);

    // Color the next-close chip based on direction
    const chipIcon = quickStats.querySelector('.stat-chip:first-child .stat-chip-icon');
    if (nextHour.close_direction === 'UP') {
        chipIcon.className = 'stat-chip-icon up';
        chipIcon.innerHTML = '<i class="fas fa-arrow-up"></i>';
    } else {
        chipIcon.className = 'stat-chip-icon down';
        chipIcon.innerHTML = '<i class="fas fa-arrow-down"></i>';
    }

    // Render Table & Chart
    renderTable(predictions);
    renderChart(historical, predictions, ticker);
}

/**
 * Render the forecast table
 */
function renderTable(predictions) {
    forecastBody.innerHTML = '';
    
    predictions.forEach(pred => {
        const row = document.createElement('tr');
        const date = new Date(pred.datetime);
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const dayStr = date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });

        row.innerHTML = `
            <td><span class="time-primary">${timeStr}</span> <span class="time-secondary">${dayStr}</span></td>
            <td>₹${pred.pred_open.toFixed(2)}</td>
            <td><strong>₹${pred.pred_close.toFixed(2)}</strong></td>
            <td><span class="trend-badge trend-${pred.close_direction.toLowerCase()}">${pred.close_direction}</span></td>
        `;
        forecastBody.appendChild(row);
    });
}

/**
 * Render Chart.js visualization with historical and prediction data
 */
function renderChart(historical, predictions, ticker) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    // Process labels — shorter format for cleaner chart
    const histLabels = historical.map(p => {
        const d = new Date(p.datetime);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });
    
    const predLabels = predictions.map(p => {
        const d = new Date(p.datetime);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    });

    const labels = [...histLabels, ...predLabels];
    const nullHist = new Array(historical.length - 1).fill(null);
    const nullPred = new Array(predictions.length).fill(null);
    
    // Process data arrays
    const histCloseData = historical.map(p => p.close);
    const histOpenData  = historical.map(p => p.open);
    const predCloseData = [...nullHist, historical[historical.length - 1].close, ...predictions.map(p => p.pred_close)];
    const predOpenData  = [...nullHist, historical[historical.length - 1].open,  ...predictions.map(p => p.pred_open)];

    // Gradients
    const histGradient = ctx.createLinearGradient(0, 0, 0, 400);
    histGradient.addColorStop(0, 'rgba(129, 140, 248, 0.3)');
    histGradient.addColorStop(1, 'rgba(129, 140, 248, 0.0)');

    const predCloseGradient = ctx.createLinearGradient(0, 0, 0, 400);
    predCloseGradient.addColorStop(0, 'rgba(56, 189, 248, 0.25)');
    predCloseGradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Past Close',
                    data: [...histCloseData, ...nullPred],
                    borderColor: '#818cf8',
                    backgroundColor: histGradient,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: '#818cf8',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    fill: true,
                    tension: 0.35
                },
                {
                    label: 'Past Open',
                    data: [...histOpenData, ...nullPred],
                    borderColor: 'rgba(129, 140, 248, 0.4)',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [3, 3],
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: '#818cf8',
                    fill: false,
                    tension: 0.35
                },
                {
                    label: 'Predicted Close',
                    data: predCloseData,
                    borderColor: '#38bdf8',
                    backgroundColor: predCloseGradient,
                    borderWidth: 2.5,
                    borderDash: [6, 4],
                    pointRadius: 3,
                    pointBackgroundColor: '#0f172a',
                    pointBorderColor: '#38bdf8',
                    pointBorderWidth: 2,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#38bdf8',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    fill: true,
                    tension: 0.35
                },
                {
                    label: 'Predicted Open',
                    data: predOpenData,
                    borderColor: '#f59e0b',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 4],
                    pointRadius: 2,
                    pointBackgroundColor: '#0f172a',
                    pointBorderColor: '#f59e0b',
                    pointBorderWidth: 2,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: '#f59e0b',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                    fill: false,
                    tension: 0.35
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.95)',
                    titleColor: '#94a3b8',
                    bodyColor: '#f8fafc',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 14,
                    cornerRadius: 10,
                    titleFont: { size: 11 },
                    bodyFont: { size: 13, weight: '600' },
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ₹${context.parsed.y?.toFixed(2) || '--'}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)', drawBorder: false },
                    ticks: { 
                        color: '#64748b', 
                        font: { size: 10 },
                        maxRotation: 45,
                        minRotation: 45,
                        maxTicksLimit: 14
                    }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)', drawBorder: false },
                    ticks: { 
                        color: '#64748b', 
                        font: { size: 11 },
                        callback: function(value) {
                            return '₹' + value.toFixed(0);
                        }
                    }
                }
            }
        }
    });
}

/**
 * Export predictions to CSV
 */
function exportCSV() {
    if (!lastPredictions) {
        showToast('No predictions to export. Run a prediction first.', 'error');
        return;
    }

    const { ticker, predictions } = lastPredictions;
    const headers = ['DateTime', 'Pred Open', 'Pred Close', 'Direction'];
    const rows = predictions.map(p => [
        p.datetime,
        p.pred_open.toFixed(2),
        p.pred_close.toFixed(2),
        p.close_direction
    ]);

    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${ticker}_forecast_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('CSV exported successfully!', 'success');
}

function showLoader(show, text = 'Processing...') {
    if (show) {
        loader.classList.remove('hidden');
        loaderText.textContent = text;
        predictBtn.disabled = true;
    } else {
        loader.classList.add('hidden');
        predictBtn.disabled = false;
    }
}

/**
 * Show a toast notification
 */
function showToast(message, type = 'error') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'error' ? 'fa-exclamation-circle' : 'fa-check-circle';
    toast.innerHTML = `
        <i class="fas ${icon}"></i>
        <span class="toast-msg">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()"><i class="fas fa-times"></i></button>
    `;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.35s ease-in forwards';
        setTimeout(() => toast.remove(), 350);
    }, 5000);
}

/**
 * Save prediction to local storage history
 */
function saveToHistory(ticker, data) {
    let history = JSON.parse(localStorage.getItem('quantEdgeHistory') || '[]');
    
    if (!data || !data.predictions || data.predictions.length === 0) return;
    
    const newEntry = {
        ticker: ticker,
        date: new Date().toISOString(),
        predictedClose: data.predictions[0].pred_close,
        direction: data.predictions[0].close_direction
    };
    
    // Add to beginning
    history.unshift(newEntry);
    
    // Keep only last 20 entries
    if (history.length > 20) history = history.slice(0, 20);
    
    localStorage.setItem('quantEdgeHistory', JSON.stringify(history));
}

/**
 * Render history in the modal
 */
function renderHistory() {
    const history = JSON.parse(localStorage.getItem('quantEdgeHistory') || '[]');
    if (history.length === 0) {
        historyList.innerHTML = '<p class="empty-state">No past predictions saved on this browser.</p>';
        return;
    }

    historyList.innerHTML = history.map(item => {
        const dateObj = new Date(item.date);
        const dateStr = dateObj.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) + ' ' + 
                        dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                        
        const trendClass = item.direction === 'UP' ? 'trend-up' : 'trend-down';
        const trendIcon = item.direction === 'UP' ? 'fa-arrow-up' : 'fa-arrow-down';
        
        return `
            <div class="card" style="margin-bottom: 12px; padding: 12px 15px; display: flex; justify-content: space-between; align-items: center; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; cursor: pointer; transition: all 0.2s ease;" onclick="loadHistoryItem('${item.ticker}')" onmouseover="this.style.borderColor='var(--primary)'" onmouseout="this.style.borderColor='rgba(255,255,255,0.08)'">
                <div style="display:flex; flex-direction:column; gap:4px;">
                    <strong style="color:var(--text-primary); font-size:16px;">${item.ticker}</strong>
                    <span style="font-size: 12px; color: var(--text-secondary);"><i class="far fa-clock"></i> ${dateStr}</span>
                </div>
                <div style="text-align: right; display:flex; flex-direction:column; gap:4px; align-items:flex-end;">
                    <span style="font-weight: 600; font-size: 15px;">₹${item.predictedClose.toFixed(2)}</span>
                    <span class="trend-badge ${trendClass}" style="font-size: 11px; padding: 2px 6px;">
                        <i class="fas ${trendIcon}" style="font-size:10px;"></i> ${item.direction}
                    </span>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Load a ticker from history
 */
window.loadHistoryItem = function(ticker) {
    tickerInput.value = ticker;
    historyModal.classList.add('hidden');
    handlePrediction();
};
