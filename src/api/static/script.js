// Interactive JavaScript with Animations
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsCard = document.getElementById('results');
    const sampleDataBtn = document.getElementById('sampleDataBtn');

    // Sample data for demonstration (benign case)
    const sampleData = {
        mean_radius: 12.32,
        mean_texture: 12.39,
        mean_perimeter: 78.85,
        mean_area: 464.1,
        mean_smoothness: 0.10280,
        mean_compactness: 0.06981,
        mean_concavity: 0.03987,
        mean_concave_points: 0.03700,
        mean_symmetry: 0.1959,
        mean_fractal_dimension: 0.05955,
        radius_error: 0.2360,
        texture_error: 0.6656,
        perimeter_error: 1.670,
        area_error: 17.43,
        smoothness_error: 0.008045,
        compactness_error: 0.011800,
        concavity_error: 0.01683,
        concave_points_error: 0.012410,
        symmetry_error: 0.01924,
        fractal_dimension_error: 0.002248,
        worst_radius: 13.50,
        worst_texture: 15.64,
        worst_perimeter: 86.97,
        worst_area: 549.1,
        worst_smoothness: 0.1385,
        worst_compactness: 0.1266,
        worst_concavity: 0.12420,
        worst_concave_points: 0.09391,
        worst_symmetry: 0.2827,
        worst_fractal_dimension: 0.06771
    };

    // Load sample data
    sampleDataBtn.addEventListener('click', function() {
        Object.keys(sampleData).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = sampleData[key];
                // Add animation effect
                input.style.transform = 'scale(1.02)';
                setTimeout(() => {
                    input.style.transform = 'scale(1)';
                }, 200);
            }
        });

        // Show success message
        showNotification('Sample data loaded successfully!', 'success');
    });

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;

        // Show loading state
        submitBtn.innerHTML = '<span class="loading"></span> Analyzing...';
        submitBtn.disabled = true;

        try {
            // Collect form data
            const formData = new FormData(form);
            const features = [];

            // Order matters for the model - match sklearn breast cancer dataset order
            const featureOrder = [
                'mean radius', 'mean texture', 'mean perimeter', 'mean area',
                'mean smoothness', 'mean compactness', 'mean concavity',
                'mean concave points', 'mean symmetry', 'mean fractal dimension',
                'radius error', 'texture error', 'perimeter error', 'area error',
                'smoothness error', 'compactness error', 'concavity error',
                'concave points error', 'symmetry error', 'fractal dimension error',
                'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                'worst smoothness', 'worst compactness', 'worst concavity',
                'worst concave points', 'worst symmetry', 'worst fractal dimension'
            ];

            featureOrder.forEach(feature => {
                const inputName = feature.replace(/ /g, '_');
                const value = parseFloat(formData.get(inputName));
                features.push(value);
            });

            // Make prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ features: features })
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();

            // Display results with animation
            displayResults(result);

        } catch (error) {
            console.error('Error:', error);
            showNotification('An error occurred during analysis. Please try again.', 'error');
        } finally {
            // Reset button
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });

    function displayResults(result) {
        const predictionResult = document.getElementById('predictionResult');
        const probabilityResult = document.getElementById('probabilityResult');
        const riskBar = document.getElementById('riskBar');
        const recommendation = document.getElementById('recommendation');

        // Determine result
        const isPositive = result.prediction === 1; // 1 = malignant, 0 = benign
        const probabilityPercent = (result.probability * 100).toFixed(1);

        // Update results
        predictionResult.textContent = isPositive ? 'High Risk (Malignant)' : 'Low Risk (Benign)';
        predictionResult.className = `metric-value ${isPositive ? 'positive' : 'negative'}`;

        probabilityResult.textContent = `${probabilityPercent}%`;

        // Animate progress bar
        riskBar.style.width = '0%';
        riskBar.style.backgroundColor = isPositive ? '#ef4444' : '#10b981';

        setTimeout(() => {
            riskBar.style.width = `${probabilityPercent}%`;
        }, 500);

        // Set recommendation
        if (isPositive) {
            recommendation.textContent = '⚠️ High risk detected. Please consult with a healthcare professional immediately for further evaluation and screening.';
            recommendation.className = 'recommendation positive';
        } else {
            recommendation.textContent = '✅ Low risk detected. Continue with regular check-ups and maintain a healthy lifestyle.';
            recommendation.className = 'recommendation negative';
        }

        // Show results with animation
        resultsCard.style.display = 'block';
        resultsCard.style.animation = 'slideInFromBottom 0.8s ease-out';

        // Scroll to results
        setTimeout(() => {
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 300);
    }

    function showNotification(message, type) {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());

        // Create notification
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        // Style notification
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.padding = '15px 20px';
        notification.style.borderRadius = '8px';
        notification.style.boxShadow = '0 10px 25px rgba(0,0,0,0.1)';
        notification.style.zIndex = '1000';
        notification.style.fontWeight = '500';
        notification.style.animation = 'slideInFromRight 0.5s ease-out';

        if (type === 'success') {
            notification.style.backgroundColor = '#10b981';
            notification.style.color = 'white';
        } else if (type === 'error') {
            notification.style.backgroundColor = '#ef4444';
            notification.style.color = 'white';
        } else {
            notification.style.backgroundColor = '#f59e0b';
            notification.style.color = 'white';
        }

        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutToRight 0.5s ease-out';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 500);
        }, 5000);
    }

    // Add CSS animations for notifications
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInFromRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOutToRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Add input validation and visual feedback
    const inputs = form.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.value < 0) {
                this.value = 0;
                showNotification('Values cannot be negative', 'error');
            }
        });

        input.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
        });

        input.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });
    });

    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add page load animation
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.8s ease-out';
        document.body.style.opacity = '1';
    }, 100);
});