import chromadb, os
client = chromadb.PersistentClient(path="vectorDB")
col = client.get_or_create_collection("personal_info")
docs = col.get()["documents"]
for doc in docs:
    print(doc.split('\n')[0].strip())