from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from flask_mail import Message
import jwt
import datetime
import os
import re
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Blueprint & DB
auth_bp = Blueprint('auth', __name__)
client = MongoClient(os.getenv('MONGO_URI'))
db = client['tumorvision_db']
users_collection = db['users']

# REGISTRASI
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not password or not email:
        return jsonify({'message': 'Username, Email, dan Password wajib diisi'}), 400

    if not re.fullmatch(r"[^@]+@gmail\.com", email):
        return jsonify({'message': 'Hanya email @gmail.com yang diizinkan'}), 400

    if users_collection.find_one({'username': username}):
        return jsonify({'message': 'Username sudah terdaftar'}), 400

    if users_collection.find_one({'email': email}):
        return jsonify({'message': 'Email sudah terdaftar'}), 400

    hashed_password = generate_password_hash(password)
    users_collection.insert_one({
        'username': username,
        'email': email,
        'password': hashed_password,
        'history': []
    })

    return jsonify({'message': 'Registrasi berhasil! Silakan login.'}), 201

# LOGIN
@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = users_collection.find_one({'username': username})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'message': 'Login gagal'}), 401

    token = jwt.encode({
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, current_app.config['SECRET_KEY'], algorithm="HS256")

    return jsonify({'token': token})

# FORGOT PASSWORD
@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'message': 'Email harus diisi.'}), 400

    user = users_collection.find_one({'email': email})
    if not user:
        return jsonify({'message': 'Email tidak ditemukan.'}), 404

    try:
        token = jwt.encode({
            'email': email,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
        }, current_app.config['SECRET_KEY'], algorithm="HS256")

        frontend_url = os.getenv('FRONTEND_URL') or 'http://localhost:5173'
        reset_link = f"{frontend_url}/reset-password/{token}"
        print("🔗 Reset link:", reset_link)

        msg = Message(
            subject="Reset Password - BrainTumor AI",
            sender=current_app.config['MAIL_USERNAME'],
            recipients=[email],
            body=f"""Halo,

Kami menerima permintaan untuk mereset password Anda di BrainTumor AI.

Silakan klik link berikut untuk mengatur ulang password Anda:
{reset_link}

Jika Anda tidak meminta ini, abaikan saja email ini.
"""
        )

        from app import mail
        mail.send(msg)
        print("✅ Email reset terkirim ke:", email)
        return jsonify({'message': 'Link reset password dikirim ke email Anda.'}), 200

    except Exception as e:
        print("❌ Gagal kirim email reset:", e)
        return jsonify({'message': 'Gagal kirim email.'}), 500

# RESET PASSWORD
@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    token = data.get('token')
    new_password = data.get('new_password')

    if not token or not new_password:
        return jsonify({'message': 'Token dan password baru wajib diisi'}), 400

    try:
        decoded = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
        email = decoded.get('email')

        user = users_collection.find_one({'email': email})
        if not user:
            return jsonify({'message': 'User tidak ditemukan'}), 404

        hashed_password = generate_password_hash(new_password)
        users_collection.update_one({'email': email}, {'$set': {'password': hashed_password}})

        return jsonify({'message': 'Password berhasil direset'}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token kadaluarsa'}), 400
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Token tidak valid'}), 400
    except Exception as e:
        print("❌ Error reset password:", e)
        return jsonify({'message': 'Gagal reset password'}), 500

# GET HISTORY
@auth_bp.route('/history', methods=['GET'])
def get_history():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({'message': 'Token tidak ditemukan'}), 401

    token = auth_header.split(" ")[1]
    try:
        decoded = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
        username = decoded.get('username')

        user = users_collection.find_one({'username': username})
        if not user:
            return jsonify({'message': 'User tidak ditemukan'}), 404

        return jsonify({'history': user.get('history', [])}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token kadaluarsa'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Token tidak valid'}), 401

# DELETE HISTORY 
@auth_bp.route('/history', methods=['DELETE'])
def delete_history():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({'message': 'Token tidak ditemukan'}), 401

    token = auth_header.split(" ")[1]
    try:
        decoded = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=["HS256"])
        username = decoded.get('username')
        timestamp_str = request.get_json().get('timestamp')

        if not timestamp_str:
            return jsonify({'message': 'Timestamp wajib dikirim'}), 400

        timestamp = datetime.datetime.fromisoformat(timestamp_str)

        result = users_collection.update_one(
            {'username': username},
            {'$pull': {'history': {'timestamp': timestamp}}}
        )

        if result.modified_count == 0:
            return jsonify({'message': 'Data tidak ditemukan'}), 404

        return jsonify({'message': 'Riwayat klasifikasi berhasil dihapus'}), 200

    except Exception as e:
        print("❌ Gagal hapus history:", e)
        return jsonify({'message': 'Gagal menghapus riwayat klasifikasi'}), 500
