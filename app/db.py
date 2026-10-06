import os
import pymongo
from datetime import datetime
from bson import ObjectId

MONGO_URI = os.getenv(
    "MONGO_URI", 
    "mongodb+srv://sumansuman207127_db_user:spUW34jBr4CYgUJK@cataract.rgyl7fe.mongodb.net/?retryWrites=true&w=majority&appName=cataract"
)
DB_NAME = os.getenv("DB_NAME", "cataract_db")

client = None
db = None

def get_database():
    global client, db
    if db is None:
        try:
            client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
            db = client[DB_NAME]
            # Verify connection
            client.admin.command('ping')
            print(f"[MongoDB Atlas] Successfully connected to database '{DB_NAME}'")
            init_indexes(db)
        except Exception as e:
            print(f"[MongoDB Atlas Warning] Connection error: {e}. Fallback in-memory/retries active.")
            db = None
    return db

def init_indexes(database):
    try:
        # Create indexes on patients collection
        database.patients.create_index([("id", pymongo.ASCENDING)], unique=True, sparse=True)
        database.patients.create_index([("patient_id", pymongo.ASCENDING)], sparse=True)
        database.patients.create_index([("asha_karmi_id", pymongo.ASCENDING)])
        database.patients.create_index([("location.state", pymongo.ASCENDING)])
        database.patients.create_index([("location.district", pymongo.ASCENDING)])
        database.patients.create_index([("cataract_status", pymongo.ASCENDING)])
        database.patients.create_index([("created_at", pymongo.DESCENDING)])
        
        # Create index on users collection
        database.users.create_index([("email", pymongo.ASCENDING)], unique=True, sparse=True)
        database.users.create_index([("mobile", pymongo.ASCENDING)], unique=True, sparse=True)
        database.users.create_index([("asha_id", pymongo.ASCENDING)], unique=True, sparse=True)
        print("[MongoDB] Indexes successfully initialized on patients and users collections.")
    except Exception as e:
        print(f"[MongoDB Index Warning] {e}")

def serialize_doc(doc):
    """Converts MongoDB document to JSON-serializable dict"""
    if not doc:
        return None
    res = dict(doc)
    if "_id" in res:
        res["_id"] = str(res["_id"])
    return res

def serialize_list(docs):
    return [serialize_doc(d) for d in docs]
