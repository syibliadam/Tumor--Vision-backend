import requests
import os

def send_reset_email(to_email, reset_link):
    api_key = os.getenv("BREVO_API_KEY")
    sender_email = os.getenv("MAIL_USERNAME")  # Gunakan email yang terdaftar & diverifikasi di Brevo
    sender_name = "BrainTumor AI"

    if not api_key or not sender_email:
        print("❌ API Key atau Sender Email tidak ditemukan di .env")
        return 500, "Missing API Key or Sender Email"

    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "sender": {
            "name": sender_name,
            "email": sender_email
        },
        "to": [
            { "email": to_email }
        ],
        "subject": "Reset Password - BrainTumor AI",
        "htmlContent": f"""
        <p>Halo,</p>
        <p>Kami menerima permintaan untuk mereset password Anda di <strong>BrainTumor AI</strong>.</p>
        <p>Silakan klik link berikut untuk mengatur ulang password Anda:</p>
        <p><a href="{reset_link}" target="_blank">{reset_link}</a></p>
        <br/>
        <p>Jika Anda tidak meminta ini, silakan abaikan email ini.</p>
        """
    }

    print("📤 Sending email to:", to_email)
    print("🔗 Reset Link:", reset_link)
    print("📨 Payload:", data)

    response = requests.post(url, headers=headers, json=data)

    print("✅ Status:", response.status_code)
    print("📩 Response:", response.text)

    return response.status_code, response.text
