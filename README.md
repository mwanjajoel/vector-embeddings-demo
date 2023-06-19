# Vector Embeddings Demo
This repo contains source code that shows a demo of how vector embeddings can help in finding similar questions from a FAQ list. The demo is based on the [Sentence Transformer](https://huggingface.co/sentence-transformers) from HuggingFace. 

## What are Vector Embeddings?
An embedding is a numerical representation of a piece of information, for example, text, documents, images, audio, etc. The representation captures the semantic meaning of what is being embedded, making it robust for many industry applications.

Embeddings are not limited to text! You can also create an embedding of an image (for example, a list of 384 numbers) and compare it with a text embedding to determine if a sentence describes the image. This concept is under powerful systems for image search, classification, description, and more!

*"[...] once you understand this ML multitool (embedding), you'll be able to build everything from search engines to recommendation systems to chatbots and a whole lot more. You don't have to be a data scientist with ML expertise to use them, nor do you need a huge labeled dataset." - Dale Markowitz, Google Cloud.*

## Process Flow
The process flow of the demo is as follows:
1. Load the FAQ list and the question to be matched.
2. Create embeddings for the FAQ list and the question to be matched.
3. Calculate the cosine similarity between the question to be matched and the FAQ list.
4. Sort the FAQ list by the cosine similarity.
5. Return the top 5 questions from the FAQ list.

## How to run the demo?
1. Clone the repo using the following command 
```
git clone https://github.com/mwanjajoel/vector-embeddings-demo.git
```

2. Create a virtual environment and install the dependencies
```
pip install -r requirements.txt
```

3. Run the demo
```
python app.py
```

4. Run the LangChain version
```
python chat.py
```

## References
- [Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings)
- [Supabase Vector Kit](https://supabase.com/vector)
- [Sentence Transformer](https://huggingface.co/sentence-transformers)
- [Sentence Transformer Documentation](https://www.sbert.net/index.html)
- [LangChain](https://python.langchain.com/docs/get_started)

## Author
- [Joel Mwanja](https://github.com/mwanjajoel)









