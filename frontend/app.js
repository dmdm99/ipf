// כתובת ברירת מחדל; ניתן לשנות ב-<details>
const state = {
    apiBase: 'http://localhost:8010',
    file: null,
};

const $ = (sel) => document.querySelector(sel);

const fileInput     = $('#fileInput');
const sendBtn       = $('#sendBtn');
const statusEl      = $('#status');
const previewWrap   = $('#previewWrap');
const previewImg    = $('#previewImg');
const resultsCard   = $('#results');
const hasArmamentEl = $('#hasArmament');
const totalEl       = $('#totalDetections');
const perClassEl    = $('#perClass');

const apiBaseInput  = $('#apiBase');
const returnImageCb = $('#returnImage');
const returnBoxesCb = $('#returnBoxes');

// Sync API base from UI
apiBaseInput.addEventListener('input', () => {
    state.apiBase = apiBaseInput.value.trim();
});

// קבלת קובץ מהמשתמש + תצוגה מקדימה
fileInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (!file) {
        state.file = null;
        previewWrap.classList.add('hidden');
        sendBtn.disabled = true;
        return;
    }
    state.file = file;
    // תצוגה מקדימה
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewWrap.classList.remove('hidden');
    sendBtn.disabled = false;
});

// שליחה ל-API
sendBtn.addEventListener('click', async () => {
    if (!state.file) return;

    // ניקוי תוצאות קודמות
    resultsCard.classList.add('hidden');
    statusEl.textContent = 'מעלה תמונה ושולח ל-API...';

    // בניית ה-FormData
    const fd = new FormData();
    fd.append('file', state.file);
    // אפשר להוסיף פרמטרים לציור/קופסאות – ברירת מחדל false כדי לשמור על תשובה קלה
    const params = new URLSearchParams();
    params.set('return_image', returnImageCb.checked ? 'true' : 'false');
    params.set('return_boxes', returnBoxesCb.checked ? 'true' : 'false');

    const url = `${state.apiBase}/predict?${params.toString()}`;

    try {
        const res = await fetch(url, {
            method: 'POST',
            body: fd,
        });

        if (!res.ok) {
            const text = await res.text();
            throw new Error(`API החזיר ${res.status}: ${text}`);
        }

        const data = await res.json();
        renderResults(data);
        statusEl.textContent = 'הצלחה ✔️';
    } catch (err) {
        console.error(err);
        statusEl.textContent = `שגיאה: ${err.message}`;
    }
});

// הצגת תוצאות רלוונטיות בלבד
function renderResults(data) {
    // data = { has_armament, total_detections, per_class, ... }

    // יש חימוש?
    hasArmamentEl.classList.remove('ok', 'alert');
    if (data.has_armament) {
        hasArmamentEl.textContent = 'כן';
        hasArmamentEl.classList.add('alert'); // אדום מושך עין
    } else {
        hasArmamentEl.textContent = 'לא';
        hasArmamentEl.classList.add('ok');    // ירוק רגוע
    }

    // סה"כ זיהויים
    totalEl.textContent = Number(data.total_detections ?? 0);

    // פירוט לפי סוג
    perClassEl.innerHTML = '';
    const perClass = data.per_class || {};
    const entries = Object.entries(perClass);
    if (entries.length === 0) {
        const li = document.createElement('li');
        li.textContent = '—';
        perClassEl.appendChild(li);
    } else {
        for (const [cls, count] of entries) {
            const li = document.createElement('li');
            li.textContent = `${cls}: ${count}`;
            perClassEl.appendChild(li);
        }
    }

    resultsCard.classList.remove('hidden');
}
