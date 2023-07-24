import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# Function to preprocess data for sequences
def prepare_data_for_sequences(df, num_instances_per_sequence):
    sequences = []
    labels = []

    for i in range(0, len(df) - num_instances_per_sequence + 1, num_instances_per_sequence):
        sequence_data = df[features].iloc[i:i+num_instances_per_sequence].values
        label = df[target].iloc[i+num_instances_per_sequence-1]
        sequences.append(sequence_data)
        labels.append(label)

    return np.array(sequences), np.array(labels)

# Function to predict stress level for new instance sequences
def predict_stress(model, scaler, person_data_sequence):
    person_data_scaled = scaler.transform(person_data_sequence.reshape(-1, num_features))
    prediction = model.predict(person_data_scaled[np.newaxis, :, :])
    return int(prediction[0, -1] > 0.5)

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
df = pd.read_csv('C:/Users/OK/dfclean.csv')

# Feature selection and target variable
features = ['Gender', 'Age', 'Bmi', 'Temperature', 'Pulse rate']
target = 'Label'

# Prepare data for sequences of 10 instances each
num_instances_per_sequence = 10
X, y = prepare_data_for_sequences(df, num_instances_per_sequence)

# Data standardization
scaler = StandardScaler()
scaler.fit(X.reshape(-1, X.shape[-1]))

# Load the trained LSTM model (replace 'path_to_your_trained_model' with the actual path to your model)
model = tf.keras.models.load_model('C:/Users/OK/trained_model.h5')

# Streamlit app
def get_session_state():
    if 'state' not in st.session_state:
        st.session_state.state = {
            'instance_num': 1,
            'gender_list': [],
            'age_list': [],
            'bmi_list': [],
            'temperature_list': [],
            'pulse_rate_list': []
        }
    return st.session_state.state

# Function to handle rerun after Next Instance button is clicked
def handle_rerun():
    st.experimental_rerun()

# Streamlit app
def main():
    st.title('Stress Prediction using LSTM')

    # Get or initialize session state
    state = get_session_state()

    # Input form for user to enter data for 1 instance
    st.write(f'Enter data for instance {state["instance_num"]}:')
    gender = st.number_input('Gender (0 for male, 1 for female)')
    age = st.number_input('Age')
    bmi = st.number_input('BMI')
    temperature = st.number_input('Temperature')
    pulse_rate = st.number_input('Pulse Rate')

    # Button to add the instance data
    if st.button('Add Instance'):
        # Append user input data to the lists
        state['gender_list'].append(gender)
        state['age_list'].append(age)
        state['bmi_list'].append(bmi)
        state['temperature_list'].append(temperature)
        state['pulse_rate_list'].append(pulse_rate)

        # Display entered data as an array
        st.write(f'Entered Data for instance {state["instance_num"]}:')
        entered_data = pd.DataFrame({
            'Gender': state['gender_list'],
            'Age': state['age_list'],
            'Bmi': state['bmi_list'],
            'Temperature': state['temperature_list'],
            'Pulse Rate': state['pulse_rate_list']
        })
        st.write(entered_data)

        # If 10 instances are entered, make the prediction
        if len(state['gender_list']) == 10:
            # Prepare user input data as a DataFrame
            user_data = pd.DataFrame({
                'Gender': state['gender_list'],
                'Age': state['age_list'],
                'Bmi': state['bmi_list'],
                'Temperature': state['temperature_list'],
                'Pulse rate': state['pulse_rate_list']
            })

            # Process data for sequences of 10 instances each
            X_user, _ = prepare_data_for_sequences(user_data, num_instances_per_sequence)

            # Make prediction using the model
            predicted_stress_level = predict_stress(model, scaler, X_user[0])

            # Display the prediction
            st.write(f'Predicted Stress Level: {predicted_stress_level}')

            # Clear the lists for the next set of 10 instances
            state['instance_num'] = 1
            state['gender_list'].clear()
            state['age_list'].clear()
            state['bmi_list'].clear()
            state['temperature_list'].clear()
            state['pulse_rate_list'].clear()

    # Button to reset the input and clear the lists
    if st.button('Reset Input'):
        state['instance_num'] = 1
        state['gender_list'].clear()
        state['age_list'].clear()
        state['bmi_list'].clear()
        state['temperature_list'].clear()
        state['pulse_rate_list'].clear()

    # Button to move to the next instance
    if st.button('Next Instance'):
        if len(state['gender_list']) < 10:
            state['instance_num'] += 1
            handle_rerun()
        else:
            st.write("Data for all instances has been entered. Click 'Reset Input' to start again.")

if __name__ == '__main__':
    main()
