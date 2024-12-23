# Customer Churn Prediction App 📊

This **Customer Churn Prediction App** is built using **Streamlit** and a pre-trained TensorFlow model. It predicts the likelihood of a customer churning based on various demographic and account-related inputs.

Access the app in your browser at [ann-predict-application](https://ann-predict-application.streamlit.app/).

---

## Features 🚀
- **Interactive User Interface**: Easy-to-use interface for inputting customer details.
- **Real-time Predictions**: Provides a customer churn probability based on user inputs.
- **Intuitive Results**: Displays predictions clearly with conditional formatting for easy interpretation.
- **Advanced Preprocessing**: Utilizes encoders and scalers for preprocessing customer data seamlessly.

---

## Tech Stack 🛠️
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: TensorFlow (Keras), Scikit-learn
- **Programming Language**: Python
- **Libraries Used**:
  - `numpy`, `pandas`: Data manipulation and processing
  - `pickle`: Serialization and deserialization of models and encoders
  - `streamlit`: Web application framework
  - `tensorflow`: Model training and predictions

---

## How It Works 🔍

### Input Customer Details:
- The user provides inputs like **geography**, **gender**, **age**, **balance**, **credit score**, etc., through a clean UI.

### Data Preprocessing:
- Inputs are encoded and scaled using pre-trained encoders and scalers to match the format required by the model.

### Model Prediction:
- The app uses a TensorFlow model (`model.h5`) to predict the churn probability.
- Predictions are displayed along with helpful interpretations.

---

## Installation Steps ⚙️

1. **Clone the repository**:
   ```bash
   git clone https://github.com/viswadarshan-024/DL-ANN-application.git
   cd churn-prediction-app
   ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Place the required files in the root directory**:
   - `model.h5`: Trained TensorFlow model.
   - `label_encoder_gender.pkl`: Label encoder for the gender feature.
   - `onehot_encoder_geo.pkl`: One-hot encoder for the geography feature.
   - `scaler.pkl`: Scaler for data normalization.

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```
---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the app. For major changes, please open an issue first to discuss what you would like to change.

---

## License 📝
This project is licensed under the [GPL-3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
