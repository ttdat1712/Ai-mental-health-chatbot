from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from configuration import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

def connect_to_mongodb():
    print("Connecting to MongoDB...")
    try:
        mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        mongo_client.admin.command('ping')
        print("Pinged MongoDB deployment. Successfully connected!")
        db = mongo_client[MONGO_DB_NAME]
        collection_history = db[MONGO_COLLECTION_NAME]
        return collection_history
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

collection_history = connect_to_mongodb()

def save_message(conversation_id, role, text):
    if collection_history is None:
        print("MongoDB collection is not available. Skipping save message.")
        return
    if not conversation_id or not text:
        print("Error: Invalid conversation_id or text for saving!")
        return
    try:
        message = {"role": role, "text": text, "timestamp": datetime.utcnow()}
        collection_history.update_one(
            {"conversation_id": conversation_id},
            {"$push": {"messages": message}},
            upsert=True
        )
    except Exception as e:
        print(f"Error saving message: {e}")

def get_conversation_history(conversation_id, max_history=3):
    if collection_history is None:
        print("MongoDB collection is not available. Returning empty history.")
        return []
    if not conversation_id:
        print("Error: Invalid conversation_id for getting history!")
        return []
    try:
        result = collection_history.find_one(
            {"conversation_id": conversation_id},
            {"_id": 0, "messages": {"$slice": -max_history}}
        )
        return result["messages"] if result and "messages" in result else []
    except Exception as e:
        print(f"Error getting history: {e}")
        return []

def delete_conversation(conversation_id):
    if collection_history is None:
        print("MongoDB collection is not available. Skipping delete conversation.")
        return False
    if not conversation_id:
        print("Error: Invalid conversation_id for deletion!")
        return False
    try:
        result = collection_history.delete_one({"conversation_id": conversation_id})
        success = result.deleted_count > 0
        print(f"Deleted conversation: {conversation_id}" if success else f"⚠ Conversation not found: {conversation_id}")
        return success
    except Exception as e:
        print(f"Error deleting conversation: {e}")
        return False