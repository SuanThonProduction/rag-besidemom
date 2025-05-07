from sentence_transformers import SentenceTransformer
import psycopg2
import ollama
import requests


embedder = SentenceTransformer("BAAI/bge-m3")


conn = psycopg2.connect(
    dbname="mydb",
    user="admin",
    password="1234",
    host="localhost",
    port="5432")
cur = conn.cursor()
# cur.execute("""
# CREATE TABLE IF NOT EXISTS documents (
# id SERIAL PRIMARY KEY,
# connect TEXT,
# embedding vector(1024)
# )
# """)
# conn.commit()
# cur.close()
# conn.close()

def add_document(text):
    
    embedding = embedder.encode(text).tolist()
    cur.execute(
        "INSERT INTO documents (connect, embedding) VALUES (%s, %s)",
        (text, embedding),
    )
    conn.commit()

document = ["1.ลูกกินนมแม่ แล้วถ่ายบ่อย ผิดปกติหรือไม่ ?เป็นปกติค่ะ เนื่องจากในนมแม่ประกอบด้วยโปรตีนที่ย่อยง่าย และในน้ำนมแม่ส่วนหน้า ประกอบด้วยน้ำตาลแลคโตส และโอลิโกแซคคาไรด์  ซึ่งช่วยส่งเสริมจุลินทรีย์ชนิดดีในลำไส้ รวมถึงฮอร์โมนพรอสตาแกลนดินที่กระตุ้นการเคลื่อนตัวของลำไส้ ดังนั้น ทารกที่กินนมแม่มักถ่ายบ่อยและอุจจาระเหลวได้ ซึ่งหากอุจจาระของทารกไม่มีมูกเลือด ไม่มีกลิ่นเหม็นเน่า ร่วมกับดูดนมได้ดี ไม่มีไข้  ไม่มีซึม หรือร้องกวนผิดปกติ คุณแม่ก็ไม่ต้องกังวลใจนะคะ",
"2.ทารกควรนอนวันละกี่ชั่วโมง ทารกมักนอนประมาณ 16 - 20 ชั่วโมงต่อวัน โดยจะตื่นเพื่อกินนมทุก 2 - 3 ชั่วโมง หากลูกนอนหลับเป็นช่วง ๆ ตื่นมาดูดนมได้ดีและน้ำหนักเพิ่มตามเกณฑ์ ถือว่าปกติค่ะ",
"3.ควรอาบน้ำลูกวันละกี่ครั้ง ทารกควรอาบน้ำวันละ 1 ครั้งก็เพียงพอ ควรใช้น้ำอุ่นและอาบในห้องที่ไม่มีลมโกรก หลังอาบน้ำควรเช็ดตัวให้แห้งทันทีเพื่อป้องกันการหนาวเย็น",
"4.ลูกสะอึกบ่อย เกิดจากอะไร ต้องทำอย่างไร การสะอึกในทารกเป็นเรื่องปกติ เพราะกระบังลมยังพัฒนาไม่สมบูรณ์ หากลูกสะอึก อาจจับเรอหรือให้ดูดนมเพื่อช่วยบรรเทาการสะอึก แต่ถ้าสะอึกนานหรือบ่อยจนรบกวนการกินนม ควรปรึกษาแพทย์ค่ะ",
"5.ควรจับเรอลูกทุกครั้งหลังให้นมไหม ใช่ค่ะ ควรจับลูกเรอทุกครั้งหลังให้นม เพื่อป้องกันอาการท้องอืดหรือแหวะนม โดยจับลูกพาดบ่า ลูบหลังเบาๆ หรืออุ้มในท่านั่งประมาณ 10 - 15 นาที",
"6.ควรให้นมบ่อยแค่ไหน ทารกคลอดก่อนกำหนดมักต้องการนมบ่อยกว่าทารกที่ครบกำหนด โดยปกติควรให้นมทุก 2 - 3 ชั่วโมง หรือประมาณวันละ 8 - 12 ครั้ง หมั่นสังเกตสัญญาณหิว เช่น การทำปากจุ๊บ ๆ หรือดูดนิ้วค่ะ",
"7.ควรปรับอุณหภูมิห้องอย่างไรให้เหมาะกับลูก อุณหภูมิห้องควรอยู่ที่ประมาณ 25 - 26°C และหลีกเลี่ยงลมเย็นโดยตรง เช่น จากพัดลมหรือแอร์ ควรห่มผ้าบาง ๆ ให้ลูกเพื่อรักษาความอบอุ่น แต่ระวังอย่าห่มแน่นเกินไปเพราะอาจทำให้ร้อนและไม่สบายตัว",
"8.ควรสังเกตอะไรบ้างในเรื่องการหายใจของลูก ทารกคลอดก่อนกำหนดอาจมีการหายใจเร็วหรือไม่สม่ำเสมอ หากลูกมีอาการหายใจหอบ ถี่ สีผิวเขียวคล้ำ หรือหยุดหายใจนานกว่า 10 วินาที ควรรีบพาไปพบแพทย์ทันที",
"9.ต้องติดตามพัฒนาการของลูกอย่างไร ทารกคลอดก่อนกำหนดอาจมีพัฒนาการที่ล่าช้ากว่าทารกครบกำหนดเล็กน้อย แนะนำให้คุณแม่ติดตามการประเมินพัฒนาการตามอายุปรับ (อายุจริงลบด้วยจำนวนสัปดาห์ที่เกิดก่อนกำหนด) *อยากให้มีลิงก์ไปยังโปรแกรมคำนวณอายุปรับในแอพค่ะ) โดยคุณแม่สามารถประเมินพัฒนาตามวัยและเรียนรู้วิธีการส่งเสริมพัฒนาการตามวัยได้ตามข้อมูลในลิงก์นี้ค่ะ นอกจากนี้เราขอแนะนำให้คุณแม่พาลูกไปพบแพทย์ตามนัดทุกครั้ง เพื่อประเมินพัฒนาการของลูกโดยผู้เชี่ยวชาญค่ะ",
"10.ลูกต้องได้รับวัคซีนเหมือนทารกปกติหรือไม่ ทารกคลอดก่อนกำหนดจำเป็นต้องได้รับวัคซีนพื้นฐาน เช่นเดียวกับทารกที่คลอดครบกำหนด อาจมีการปรับเพิ่มตามคำแนะนำของแพทย์ ทั้งนี้แนะนำให้คุณแม่พาลูกไปรับวัคซีนตามนัดทุกครั้ง เพื่อส่งเสริมภูมิคุ้มกันโรคให้ลูกค่ะ",
"11.ควรจัดท่านอนลูกอย่างไร แนะนำให้ลูกนอนหงายบนที่นอนที่เรียบและแน่น และหลีกเลี่ยงการนำหมอน ผ้าห่ม หรือของเล่นวางในเตียงนอน เพื่อป้องกันการอุดกั้นทางเดินหายใจ ซึ่งเป็นสาเหตุให้ทารกเกิดการเสียชีวิตแบบฉับพลันได้ ทั้งนี้คุณแม่ไม่ควรจัดท่านอนคว่ำหรือนอนตะแคงให้แก่ทารก แต่หากคุณแม่กังวลใจเกี่ยวกับรูปทรงศีรษะของลูก คุณแม่ควรดูแลลูกอย่างใกล้ชิดขณะจัดท่านอนคว่ำหรือนอนตะแคง",
"12.รู้ได้อย่างไรว่าลูกได้รับนมเพียงพอ หากทารกได้รับนมเพียงพอ คุณแม่จะสังเกตได้ว่า ทารกจะนอนหลับนาน 2 – 3 ชั่วโมงติดต่อกัน ปัสสาวะสีเหลืองอ่อน 6 – 8 ครั้งต่อวัน อุจจาระอย่างน้อย 2 ครั้งต่อวัน และมีน้ำหนักขึ้นตามเกณฑ์ ",
"13.ต้องทำอย่างไรหากลูกน้ำหนักตัวขึ้นช้า หากลูกน้ำหนักไม่เพิ่มขึ้นตามเกณฑ์ ควรปรึกษาแพทย์ทันที คุณแม่อาจต้องให้นมบ่อยขึ้นหรือปรับสูตรนมเสริมตามคำแนะนำของแพทย์ พร้อมทั้งตรวจสอบการดูดนมของลูกว่ามีประสิทธิภาพเพียงพอหรือไม่",
"14.ลูกมีอาการแหวะนมบ่อย ผิดปกติหรือไม่ ทารกคลอดก่อนกำหนดอาจมีการแหวะนมบ่อย เนื่องจากหูรูดกระเพาะอาหารยังพัฒนาไม่เต็มที่ ให้ป้อนนมทีละน้อยและจับลูกเรอหลังให้นม หากลูกยังแหวะนมบ่อย น้ำหนักตัวไม่ขึ้น หรือมีการสำลัก ควรรีบปรึกษาแพทย์ค่ะ",
"15.สามารถอุ้มลูกนาน ๆ ได้ไหม,การอุ้มลูกสามารถทำได้ตามปกติ โดยเฉพาะการอุ้มแบบ Kangaroo Care คือ การให้อุ้มให้ผิวหนังของลูกแนบตัวคุณแม่หรือคุณพ่อ ซึ่งจะช่วยให้ลูกอบอุ่น ลดความเครียด และกระตุ้นการเจริญเติบโตได้ค่ะ",
"16.เสียงหายใจครืดคราดในทารกคลอดก่อนกำหนดอาจเกิดจากน้ำมูกหรือเสมหะที่ค้างอยู่ เนื่องจากระบบหายใจยังพัฒนาไม่สมบูรณ์ หากลูกไม่มีอาการหอบเหนื่อยหรือเขียวคล้ำ ให้ดูดน้ำมูกออกเบา ๆ แต่ถ้าเสียงหายใจยังดังมากขึ้น มีอาการหอบหรือหายใจลำบาก ผิวหนังบริเวณใบหน้ามีสีเขียวหรือม่วงคล้ำ ควรรีบพาทารกไปพบแพทย์ค่ะ",
"17.ทำไมลูกดูเหมือนหลับบ่อยผิดปกติ ทารกคลอดก่อนกำหนดมักนอนหลับมากกว่าทารกคลอดครบกำหนด เพราะต้องสงวนพลังงานไว้ใช้ในการเจริญเติบโต หากลูกตื่นมากินนมและน้ำหนักเพิ่มขึ้นตามเกณฑ์ ถือว่าเป็นเรื่องปกติ แต่ถ้าลูกซึมมาก ไม่ตอบสนอง หรือไม่ยอมตื่นมากินนม ควรพบแพทย์ทันทีค่ะ",
"18.ควรดูแลลูกอย่างไรเมื่อลูกต้องการออกนอกบ้าน ควรเลื่อนการพาลูกออกนอกบ้านจนกว่าร่างกายจะแข็งแรงมากขึ้น หากจำเป็นต้องออกนอกบ้าน ควรหลีกเลี่ยงที่ชุมชนและคนแออัด เตรียมเสื้อผ้าที่อบอุ่นและหน้ากากสำหรับผู้ใหญ่ที่อุ้มลูก เพื่อป้องกันการติดเชื้อจากสิ่งแวดล้อม",
]
    


# for doc in document:
#     add_document(doc)
#     print("done")
# cur.close()
# conn.close()


def query_documents(query):
    embedding = embedder.encode(query).tolist()
    conn = psycopg2.connect(
    dbname="mydb",
    user="admin",
    password="1234",
    host="localhost",
    port="5432")
    cur = conn.cursor()
    print("query embedding", embedding)
    sql_query = """
    SELECT connect, embedding <=> %s::vector AS similarity_score
    FROM documents
    ORDER BY similarity_score
    LIMIT 5;
    """
    cur.execute(sql_query, (embedding,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

   
    



def generate_response(prompt):
    retrieved_docs = query_documents(prompt)
    context = "\n".join([doc[0] for doc in retrieved_docs])
    print("context", context)

    prompt_text = f"Answer the question based on the context provided.\n\nContext: {context}\n\nQuestion: {prompt}\nAnswer:"
    print(prompt_text)
    # Ensure the model is pulled before using it
    response = ollama.chat(model="llama3.2", messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}])

    return response["message"]["content"]

def generate_response2(prompt):
    retrieved_docs = query_documents(prompt)
    context = "\n".join([doc[0] for doc in retrieved_docs])
    print("Context:", context)

    prompt_text = f"Answer the question based on the context provided.\n\nContext: {context}\n\nQuestion: {prompt}\nAnswer:"
    print("Full prompt sent to API:\n", prompt_text)

    # Replace with your actual Typhoon API key
    api_key = "sk-titCpkply6rFBcc6326yqTJzL20JeJJ9ACekZEij4nNCpClA"
    api_url = "https://api.opentyphoon.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "typhoon-v2-70b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # Raise error if request fails
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.SSLError as ssl_error:
        print(f"SSL Error: {ssl_error}")
        return "An SSL error occurred. Please check your API endpoint and SSL configuration."
    except requests.exceptions.RequestException as req_error:
        print(f"Request Error: {req_error}")
        return "An error occurred while making the request. Please try again later."

# Example usage
query = "รู้ได้อย่างไรว่าลูกได้รับนมเพียงพอ"
print(generate_response2(query))

