#!/usr/bin/env python3
"""
Test script for RAG API with PDF upload functionality
"""

import requests
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def check_server_health():
    """Check if the server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server Status: {health_data['status']}")
            print(f"📊 Ready: {health_data['ready']}")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach server: {e}")
        print("Make sure the server is running: uvicorn rag_api:app --reload --port 8000")
        return False

def check_pdf_status():
    """Check current PDF and RAG system status"""
    try:
        response = requests.get(f"{API_BASE_URL}/pdf-status")
        if response.status_code == 200:
            status_data = response.json()
            print("\n📄 PDF Status:")
            print(f"  RAG Initialized: {status_data['rag_initialized']}")
            print(f"  Chunks Count: {status_data['chunks_count']}")
            print(f"  Default PDF Exists: {status_data['default_pdf_exists']}")
            print(f"  Embedder Loaded: {status_data['embedder_loaded']}")
            print(f"  Message: {status_data['message']}")
            return status_data['rag_initialized']
        else:
            print(f"❌ Failed to get PDF status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking PDF status: {e}")
        return False

def upload_pdf(pdf_path):
    """Upload a PDF file to the API"""
    try:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        print(f"📤 Uploading PDF: {pdf_file.name}")
        
        with open(pdf_file, 'rb') as file:
            files = {'file': (pdf_file.name, file, 'application/pdf')}
            response = requests.post(f"{API_BASE_URL}/upload-pdf", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"  Filename: {result['filename']}")
            print(f"  Chunks Created: {result['chunks_count']}")
            print(f"  Status: {result['status']}")
            print(f"  Message: {result['message']}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading PDF: {e}")
        return False

def test_chat(message="ลูกกินนมแม่แล้วถ่ายบ่อยผิดปกติหรือไม่?"):
    """Test the chat functionality"""
    try:
        print(f"\n💬 Testing chat with message: {message}")
        
        data = {
            "message": message,
            "max_tokens": 512
        }
        
        response = requests.post(f"{API_BASE_URL}/chat", json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat successful!")
            print("🤖 Bot Response:")
            print("-" * 50)
            print(result["response"])
            print("-" * 50)
            print("\n📚 Retrieved Context (first 200 chars):")
            context = result["retrieved_context"]
            print(context[:200] + "..." if len(context) > 200 else context)
            return True
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during chat: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing RAG API with PDF Upload")
    print("=" * 50)
    
    # 1. Check server health
    if not check_server_health():
        return
    
    # 2. Check current PDF status
    rag_ready = check_pdf_status()
    
    # 3. If no PDF loaded, try to upload one
    if not rag_ready:
        print("\n📤 No PDF loaded. Looking for PDF files to upload...")
        
        # Look for PDF files in current directory
        pdf_files = list(Path(".").glob("*.pdf"))
        if pdf_files:
            pdf_to_upload = pdf_files[0]  # Use the first PDF found
            print(f"Found PDF: {pdf_to_upload}")
            
            if upload_pdf(pdf_to_upload):
                rag_ready = True
        else:
            print("❌ No PDF files found in current directory")
            print("Please place a PDF file in the current directory or upload via the API")
            return
    
    # 4. Test chat functionality
    if rag_ready:
        print("\n🚀 RAG system is ready! Testing chat...")
        test_chat()
        
        # Test with additional questions
        additional_tests = [
            "ลูกเจ็บป่วยแล้วไม่ยอมกินนมแม่ ควรทำอย่างไร?",
            "วิธีการให้นมแม่ที่ถูกต้องคืออะไร?"
        ]
        
        for i, test_msg in enumerate(additional_tests, 2):
            print(f"\n--- Additional Test {i} ---")
            test_chat(test_msg)
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
