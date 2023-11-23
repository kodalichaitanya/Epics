import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load your pre-trained model
model = load_model('sleep_apnea_model.keras')

def predict_ecg(file_path):
    try:
        # Load the ECG signal data
        p_signal = np.loadtxt(file_path)
        p_signal = p_signal[:3000].reshape(1, 3000, 1)

        # Make predictions using the loaded model
        predicted_probability = model.predict(p_signal)
        predicted_class = 1 if predicted_probability > 0.5 else 0
        predicted_class_label = 'Apnea' if predicted_class == 1 else 'Normal'
        return predicted_class_label
    except Exception as e:
        return "Error: " + str(e)

def generate_pdf_report(file_path, user_details, predicted_class_label):
    pdf_report_path = 'ecg_report.pdf'
    
    with PdfPages(pdf_report_path) as pdf:
        # Create a plot of the ECG data
        p_signal = np.loadtxt(file_path)[:3000]
        plt.figure(figsize=(8, 4))
        plt.plot(p_signal, color='blue')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('ECG Signal')
        pdf.savefig()
        plt.close()
        
        # Create a report with user details and prediction result
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.text(0.1, 0.9, 'ECG Classification Report', fontsize=14)
        ax.text(0.1, 0.8, f'Name: {user_details["name"]}', fontsize=12)
        ax.text(0.1, 0.75, f'Age: {user_details["age"]}', fontsize=12)
        ax.text(0.1, 0.7, f'Gender: {user_details["gender"]}', fontsize=12)
        ax.text(0.1, 0.6, f'Mobile Number: {user_details["mobile"]}', fontsize=12)
        ax.text(0.1, 0.5, f'Predicted Class: {predicted_class_label}', fontsize=12)
        pdf.savefig()
    
    return pdf_report_path

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select ECG Signal File")
    if file_path:
        # Create a dialog to input user details
        user_details = {}
        user_dialog = tk.Toplevel(window)
        user_dialog.title("User Details")
        tk.Label(user_dialog, text="Name:").grid(row=0, column=0)
        tk.Label(user_dialog, text="Age:").grid(row=1, column=0)
        tk.Label(user_dialog, text="Gender:").grid(row=2, column=0)
        tk.Label(user_dialog, text="Mobile Number:").grid(row=3, column=0)
        
        entry_name = tk.Entry(user_dialog)
        entry_age = tk.Entry(user_dialog)
        entry_gender = tk.Entry(user_dialog)
        entry_mobile = tk.Entry(user_dialog)
        
        entry_name.grid(row=0, column=1)
        entry_age.grid(row=1, column=1)
        entry_gender.grid(row=2, column=1)
        entry_mobile.grid(row=3, column=1)
        
        def submit_user_details():
            user_details["name"] = entry_name.get()
            user_details["age"] = entry_age.get()
            user_details["gender"] = entry_gender.get()
            user_details["mobile"] = entry_mobile.get()
            
            predicted_class_label = predict_ecg(file_path)
            result_label.config(text=f"Predicted Class: {predicted_class_label}")
            
            pdf_report_path = generate_pdf_report(file_path, user_details, predicted_class_label)
            messagebox.showinfo("Report Generated", f"PDF report saved at {pdf_report_path}")
            user_dialog.destroy()
            
        submit_button = tk.Button(user_dialog, text="Submit", command=submit_user_details)
        submit_button.grid(row=4, column=0, columnspan=2)
        
def open_pdf_report():
    pdf_report_path = generate_pdf_report("example_file.txt", {'name': 'John Doe', 'age': 30, 'gender': 'Male', 'mobile': '1234567890'}, "Normal")
    import os
    os.system(f'start {pdf_report_path}')

# Create a Tkinter window
window = tk.Tk()
window.title("ECG Classification")

# Create a button to open the file dialog
open_button = tk.Button(window, text="Open ECG Signal File", command=open_file_dialog)
open_button.pack(pady=20)

# Create a label to display the prediction result
result_label = tk.Label(window, text="", font=("Helvetica", 16))
result_label.pack()

# Create a button to open PDF report
pdf_button = tk.Button(window, text="Open PDF Report", command=open_pdf_report)
pdf_button.pack(pady=10)

# Run the Tkinter main loop
window.mainloop()
