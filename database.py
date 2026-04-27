"""
Database Management for Brain Tumor Detection App
Handles user login, registration, and prediction history
"""

import sqlite3
import hashlib
from datetime import datetime
import os
from pathlib import Path

DB_PATH = "predictions_database.db"

def init_database():
    """Initialize database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_name TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email, phone, password):
    """Register new user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        hashed_password = hash_password(password)
        cursor.execute('''
            INSERT INTO users (name, email, phone, password)
            VALUES (?, ?, ?, ?)
        ''', (name, email, phone, hashed_password))
        
        conn.commit()
        conn.close()
        return True, "✅ Registration successful!"
    
    except sqlite3.IntegrityError:
        return False, "❌ Email already registered!"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def login_user(email, password):
    """Login user and return user info"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        hashed_password = hash_password(password)
        cursor.execute('''
            SELECT id, name, email, phone FROM users
            WHERE email = ? AND password = ?
        ''', (email, hashed_password))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return True, {
                'user_id': user[0],
                'name': user[1],
                'email': user[2],
                'phone': user[3]
            }
        else:
            return False, "❌ Invalid email or password"
    
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def save_prediction(user_id, image_name, predicted_class, confidence):
    """Save prediction result to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (user_id, image_name, predicted_class, confidence)
            VALUES (?, ?, ?, ?)
        ''', (user_id, image_name, predicted_class, confidence))
        
        conn.commit()
        conn.close()
        return True, "✅ Prediction saved to database!"
    
    except Exception as e:
        return False, f"❌ Error saving prediction: {str(e)}"

def get_user_predictions(user_id):
    """Get all predictions for a specific user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, image_name, predicted_class, confidence, timestamp
            FROM predictions
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
        
        predictions = cursor.fetchall()
        conn.close()
        return predictions
    
    except Exception as e:
        return []

def get_all_statistics():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total users
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        # Tumor distribution
        cursor.execute('''
            SELECT predicted_class, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_class
        ''')
        tumor_dist = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_users': total_users,
            'total_predictions': total_predictions,
            'tumor_distribution': tumor_dist
        }
    
    except Exception as e:
        return None

# Initialize database on module load
if not os.path.exists(DB_PATH):
    init_database()
