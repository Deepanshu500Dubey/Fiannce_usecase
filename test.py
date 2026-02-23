# run_with_your_data.py
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def print_response(step, response):
    """Pretty print API responses"""
    print(f"\n{'='*60}")
    print(f"📌 {step}")
    print('='*60)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

def main():
    # Step 1: Check API is running
    print("Checking API health...")
    response = requests.get(f"{BASE_URL}/health")
    print_response("API Health Check", response)
    
    # Step 2: Upload your Raw_GL_Export.csv
    print("\n" + "="*60)
    print("📤 UPLOADING YOUR Raw_GL_Export.csv")
    print("="*60)
    
    with open('Raw_GL_Export.csv', 'rb') as f:
        files = {'file': ('Raw_GL_Export.csv', f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload/gl", files=files)
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return
    
    upload_result = response.json()
    print(f"✅ Upload successful!")
    print(f"📊 Rows: {upload_result['rows']}")
    print(f"📁 Path: {upload_result['path']}")
    
    gl_file_path = upload_result['path']
    
    # Step 3: Start processing
    print("\n" + "="*60)
    print("⚙️ STARTING PIPELINE PROCESSING")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/process/full",
        params={"gl_file_path": gl_file_path}
    )
    
    if response.status_code != 200:
        print(f"❌ Processing start failed: {response.text}")
        return
    
    process_result = response.json()
    task_id = process_result['task_id']
    print(f"✅ Processing started!")
    print(f"🆔 Task ID: {task_id}")
    
    # Step 4: Monitor progress
    print("\n" + "="*60)
    print("⏳ MONITORING PROGRESS")
    print("="*60)
    
    while True:
        response = requests.get(f"{BASE_URL}/tasks/{task_id}")
        status = response.json()
        
        print(f"\rStatus: {status['status']} | Last Updated: {status.get('updated_at', 'N/A')}", end="")
        
        if status['status'] in ['completed', 'failed']:
            print("\n")
            break
        
        time.sleep(2)
    
    # Step 5: Check final result
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    if status['status'] == 'completed':
        print("✅ Processing completed successfully!")
        
        # Download processed data
        print("\n📥 Downloading results...")
        
        # Download standardized GL data
        response = requests.get(f"{BASE_URL}/data/gl/{task_id}")
        if response.status_code == 200:
            with open('processed_gl_data.csv', 'wb') as f:
                f.write(response.content)
            print("✅ Downloaded: processed_gl_data.csv")
        
        # Download anomalies report
        response = requests.get(f"{BASE_URL}/reports/{task_id}?report_type=anomalies")
        if response.status_code == 200:
            with open('anomalies_report.csv', 'wb') as f:
                f.write(response.content)
            print("✅ Downloaded: anomalies_report.csv")
        
        # Download exceptions report
        response = requests.get(f"{BASE_URL}/reports/{task_id}?report_type=exceptions")
        if response.status_code == 200:
            with open('exceptions_report.csv', 'wb') as f:
                f.write(response.content)
            print("✅ Downloaded: exceptions_report.csv")
        
        # Download variance report
        response = requests.get(f"{BASE_URL}/reports/{task_id}?report_type=variance")
        if response.status_code == 200:
            with open('variance_report.csv', 'wb') as f:
                f.write(response.content)
            print("✅ Downloaded: variance_report.csv")
        
        # Download executive narrative
        response = requests.get(f"{BASE_URL}/reports/{task_id}?report_type=narrative")
        if response.status_code == 200:
            with open('executive_narrative.txt', 'wb') as f:
                f.write(response.content)
            print("✅ Downloaded: executive_narrative.txt")
        
        print("\n📁 All files saved to current directory!")
        
    else:
        print(f"❌ Processing failed: {status.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()