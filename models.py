from flask_sqlalchemy import SQLAlchemy
from flask import Flask

db = SQLAlchemy()
# Định nghĩa các bảng



class Class(db.Model):
    __tablename__ = 'classes'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)

class Student(db.Model):
    __tablename__ = 'students'
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String, nullable=True)
    profile_picture = db.Column(db.String, nullable=True)
    profile_picture_vector = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)

class ClassSession(db.Model):
    __tablename__ = 'class_sessions'
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('classes.id'), nullable=False)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)

class Enrollment(db.Model):
    __tablename__ = 'enrollments'
    student_id = db.Column(db.String, db.ForeignKey('students.id'), primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), primary_key=True)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)

class Grade(db.Model):
    __tablename__ = 'grades'
    student_id = db.Column(db.String, db.ForeignKey('students.id'), primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('classes.id'), primary_key=True)
    final_grade = db.Column(db.Float, nullable=True)
    midterm_grade = db.Column(db.Float, nullable=True)
    attendance_grade = db.Column(db.Float, nullable=True)
    lab_grade = db.Column(db.Float, nullable=True)
    extra_grade = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)


class AttendanceRecord(db.Model):
    __tablename__ = 'attendance_records'
    student_id = db.Column(db.String, db.ForeignKey('students.id'), primary_key=True)
    class_session_id = db.Column(db.Integer, db.ForeignKey('class_sessions.id'), primary_key=True)
    status = db.Column(db.String, nullable=False)  # Sử dụng varchar thay vì enum
    created_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=True)



# Cấu hình SQLite
if __name__ == '__main__':
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///class_management.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database created")