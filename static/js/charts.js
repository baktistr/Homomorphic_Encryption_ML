/* Chart.js rendering helpers for the comparison dashboard. */

const COLORS = {
    normal: '#4A90D9',
    normalLight: 'rgba(74, 144, 217, 0.2)',
    he: '#2ECC71',
    heLight: 'rgba(46, 204, 113, 0.2)',
    encrypt: '#3498DB',
    infer: '#E67E22',
    decrypt: '#9B59B6',
    error: '#E74C3C',
};

function renderMetricsChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const labels = data.map(d => d.model.display_name);
    const ptF1 = data.map(d => d.pt_metrics ? d.pt_metrics.f1 : 0);
    const heF1 = data.map(d => d.he_metrics ? d.he_metrics.f1 : 0);
    const ptPRAUC = data.map(d => d.pt_metrics ? d.pt_metrics.pr_auc : 0);
    const hePRAUC = data.map(d => d.he_metrics ? d.he_metrics.pr_auc : 0);

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Normal F1', data: ptF1, backgroundColor: COLORS.normal },
                { label: 'HE F1', data: heF1, backgroundColor: COLORS.he },
                { label: 'Normal PR-AUC', data: ptPRAUC, backgroundColor: COLORS.normalLight, borderColor: COLORS.normal, borderWidth: 2 },
                { label: 'HE PR-AUC', data: hePRAUC, backgroundColor: COLORS.heLight, borderColor: COLORS.he, borderWidth: 2 },
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'F1 & PR-AUC: Normal vs HE' } },
            scales: { y: { beginAtZero: true, max: 1.0 } }
        }
    });
}

function renderLatencyChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const labels = data.filter(d => d.he_available).map(d => d.model.display_name);
    const encrypt = data.filter(d => d.he_available).map(d => d.he_timing.avg_encrypt_ms);
    const infer = data.filter(d => d.he_available).map(d => d.he_timing.avg_infer_ms);
    const decrypt = data.filter(d => d.he_available).map(d => d.he_timing.avg_decrypt_ms);

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Encrypt (ms)', data: encrypt, backgroundColor: COLORS.encrypt },
                { label: 'Infer (ms)', data: infer, backgroundColor: COLORS.infer },
                { label: 'Decrypt (ms)', data: decrypt, backgroundColor: COLORS.decrypt },
            ]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'HE Latency Breakdown (per sample avg)' } },
            scales: { x: { stacked: true }, y: { stacked: true, title: { display: true, text: 'ms' } } }
        }
    });
}

function renderErrorChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Scatter: per-model, show individual sample errors from he_records
    const datasets = [];
    const colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'];

    data.forEach((item, i) => {
        if (item.he_available && item.he_records && item.he_records.length > 0) {
            datasets.push({
                label: item.model.display_name,
                data: item.he_records.map((r, idx) => ({ x: idx, y: r.absolute_error })),
                backgroundColor: colors[i % colors.length],
                pointRadius: 3,
            });
        }
    });

    if (datasets.length === 0) return;

    new Chart(canvas, {
        type: 'scatter',
        data: { datasets: datasets },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'Per-Sample Absolute Error (HE vs Plaintext)' } },
            scales: {
                x: { title: { display: true, text: 'Sample Index' } },
                y: { title: { display: true, text: '|Error|' }, type: 'logarithmic' }
            }
        }
    });
}

function renderSweepChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || data.length === 0) return;

    // Group by model
    const models = [...new Set(data.map(d => d.display_name))];
    const configs = [...new Set(data.map(d => d.config_name))];
    const colors = ['#4A90D9', '#2ECC71', '#E74C3C', '#F39C12'];

    const datasets = models.map((model, i) => ({
        label: model,
        data: configs.map(cfg => {
            const row = data.find(d => d.display_name === model && d.config_name === cfg);
            return row ? row.avg_total_ms : 0;
        }),
        backgroundColor: colors[i % colors.length],
    }));

    new Chart(canvas, {
        type: 'bar',
        data: { labels: configs, datasets: datasets },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: 'Avg Total HE Time (ms) by CKKS Config' } },
            scales: { y: { title: { display: true, text: 'ms' } } }
        }
    });
}
