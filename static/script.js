let selectedSymptoms = [];

function filterSymptoms(query) {
  const chips = document.querySelectorAll('.symptom-chip');
  chips.forEach(chip => {
    const text = chip.querySelector('span').textContent.toLowerCase();
    chip.style.display = text.includes(query.toLowerCase()) ? 'block' : 'none';
  });
}

function updateSelected() {
  const checked = document.querySelectorAll('.symptom-chip input:checked');
  selectedSymptoms = Array.from(checked).map(c => c.value);
  const bar = document.getElementById('selected-bar');
  const btn = document.getElementById('predict-btn');
  const count = document.getElementById('selected-count');
  if (selectedSymptoms.length > 0) {
    bar.style.display = 'flex';
    count.textContent = selectedSymptoms.length + ' symptom(s) selected';
    btn.disabled = false;
  } else {
    bar.style.display = 'none';
    btn.disabled = true;
  }
}

function clearAll() {
  document.querySelectorAll('.symptom-chip input:checked').forEach(c => c.checked = false);
  updateSelected();
}

async function predict() {
  const btn = document.getElementById('predict-btn');
  btn.textContent = 'Analysing...';
  btn.disabled = true;

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms: selectedSymptoms })
    });
    const data = await res.json();

    document.getElementById('result-disease').textContent = data.disease;
    document.getElementById('result-confidence').textContent = 'Confidence: ' + data.confidence + '%';
    document.getElementById('result-description').textContent = data.description;

    const makeList = (id, items) => {
      const ul = document.getElementById(id);
      ul.innerHTML = items.map(i => `<li>${i}</li>`).join('');
    };
    makeList('result-medicines', data.medicines);
    makeList('result-diet', data.diet);
    makeList('result-workout', data.workout);

    document.getElementById('results-section').style.display = 'block';
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    alert('Error connecting to server. Make sure app.py is running.');
  }

  btn.textContent = 'Analyse Symptoms';
  btn.disabled = false;
}
```

---

## Step 7 — `requirements.txt`
```
flask
scikit-learn
pandas
numpy
joblib