// Global variables
const API_URL = PREDICTION_API_URL;
const formView = document.getElementById('formView');
const resultView = document.getElementById('resultView');
const predictionForm = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const backBtn = document.getElementById('backBtn');

// Form submission handler
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Clear any existing errors
    errorMessage.classList.add('hidden');
    errorText.textContent = '';

    // Gather form data
    const formData = new FormData(predictionForm);
    const customerData = {
        Gender: formData.get('gender'),
        SeniorCitizen: parseInt(formData.get('SeniorCitizen')),
        Partner: formData.get('Partner'),
        Dependents: formData.get('Dependents'),
        tenure: parseInt(formData.get('tenure')),
        PhoneService: formData.get('PhoneService'),
        MultipleLines: formData.get('MultipleLines'),
        InternetService: formData.get('InternetService'),
        OnlineSecurity: formData.get('OnlineSecurity'),
        OnlineBackup: formData.get('OnlineBackup'),
        DeviceProtection: formData.get('DeviceProtection'),
        TechSupport: formData.get('TechSupport'),
        StreamingTV: formData.get('StreamingTV'),
        StreamingMovies: formData.get('StreamingMovies'),
        Contract: formData.get('Contract'),
        PaperlessBilling: formData.get('PaperlessBilling'),
        PaymentMethod: formData.get('PaymentMethod'),
        MonthlyCharges: parseFloat(formData.get('MonthlyCharges')),
        TotalCharges: parseFloat(formData.get('TotalCharges'))
    };

    // Disable button and show loading state
    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing...';
    submitBtn.classList.add('cursor-not-allowed', 'opacity-75');

    try {
        // Make API request with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(customerData),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorData = await response.json();
            // Handle validation errors (422) which return an array of error objects
            let errorMsg = 'Prediction failed';
            if (Array.isArray(errorData.detail)) {
                errorMsg = errorData.detail.map(err => `${err.loc[err.loc.length - 1]}: ${err.msg}`).join(', ');
            } else if (typeof errorData.detail === 'string') {
                errorMsg = errorData.detail;
            }
            throw new Error(errorMsg);
        }

        const result = await response.json();
        showResult(result);

    } catch (error) {
        if (error.name === 'AbortError') {
            showError('Request timed out. Please check your connection and try again.');
        } else if (error.message.includes('fetch')) {
            showError('Unable to connect to prediction service. Please try again.');
        } else if (error.message.includes('model not found')) {
            showError('Prediction model not found. Please ensure model.pkl is in the correct location.');
        } else {
            showError(error.message || 'An unexpected error occurred. Please try again later.');
        }
    } finally {
        // Re-enable button
        submitBtn.disabled = false;
        submitBtn.textContent = 'Predict Churn Risk';
        submitBtn.classList.remove('cursor-not-allowed', 'opacity-75');
    }
});

// Show error message
function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
}

// Show result view
function showResult(data) {
    const { result, label, probability } = data;
    const isChurn = result === 1 || label === 'Churn';
    
    // Parse probability from string (e.g., "51.01%" -> 0.5101)
    const probValue = parseFloat(probability.replace('%', '')) / 100;

    // Get result elements
    const badge = document.getElementById('resultBadge');
    const title = document.getElementById('resultTitle');
    const subtitle = document.getElementById('resultSubtitle');
    const confidence = document.getElementById('resultConfidence');

    if (isChurn) {
        // High risk (will churn)
        badge.textContent = 'HIGH RISK';
        badge.className = 'inline-block px-4 py-2 rounded-full text-sm font-semibold bg-red-100 text-red-800 border border-red-300';

        title.textContent = 'Customer Will Likely Churn';
        title.className = 'text-3xl font-bold mb-2 text-red-800';

        subtitle.textContent = 'Consider retention strategies';

        confidence.textContent = (probValue * 100).toFixed(2) + '%';
        confidence.className = 'text-6xl font-bold mb-2 text-red-600';
    } else {
        // Low risk (will stay)
        badge.textContent = 'LOW RISK';
        badge.className = 'inline-block px-4 py-2 rounded-full text-sm font-semibold bg-green-100 text-green-800 border border-green-300';

        title.textContent = 'Customer Will Likely Stay';
        title.className = 'text-3xl font-bold mb-2 text-green-800';

        subtitle.textContent = 'Customer shows strong retention signals';

        confidence.textContent = ((1 - probValue) * 100).toFixed(2) + '%';
        confidence.className = 'text-6xl font-bold mb-2 text-green-600';
    }

    // Hide form, show result
    formView.classList.add('hidden');
    resultView.classList.remove('hidden');
}

// Back to form handler
backBtn.addEventListener('click', () => {
    resultView.classList.add('hidden');
    formView.classList.remove('hidden');
    // Optionally reset form
    // predictionForm.reset();
});