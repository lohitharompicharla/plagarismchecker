import os
import string
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class TextAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("Document Analyzer GUI")
        master.geometry("400x300")

        self.label = tk.Label(master, text="Document Analyzer", font=("Arial", 16, "bold"))
        self.label.pack(pady=10)

        self.select_button = tk.Button(master, text="Select Folder", command=self.select_folder)
        self.select_button.pack(pady=5)

        self.similarity_button = tk.Button(master, text="Show Similarity Heatmap", command=self.show_similarity)
        self.similarity_button.pack(pady=5)

        self.wordcloud_button = tk.Button(master, text="Generate Word Clouds", command=self.show_wordclouds)
        self.wordcloud_button.pack(pady=5)

        self.topwords_button = tk.Button(master, text="Top Frequent Words", command=self.show_top_words)
        self.topwords_button.pack(pady=5)

        self.sentiment_button = tk.Button(master, text="Sentiment Analysis", command=self.analyze_sentiment)
        self.sentiment_button.pack(pady=5)

        self.plagiarism_button = tk.Button(master, text="Detect Plagiarism", command=self.detect_plagiarism)
        self.plagiarism_button.pack(pady=5)

        self.export_button = tk.Button(master, text="Export CSV", command=self.export_csv)
        self.export_button.pack(pady=5)

        self.folder_path = ""
        self.documents = {}
        self.processed_documents = {}
        self.doc_names = []
        self.similarity_matrix = None

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if not self.folder_path:
            return

        self.documents = self.load_documents(self.folder_path)
        self.processed_documents = {filename: self.preprocess_text(content) for filename, content in self.documents.items()}
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(self.processed_documents.values())
        self.similarity_matrix = cosine_similarity(doc_vectors)
        self.doc_names = list(self.processed_documents.keys())
        messagebox.showinfo("Success", f"{len(self.doc_names)} files loaded.")

    def load_documents(self, folder_path):
        documents = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        documents[filename] = file.read()
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        return documents

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    def show_similarity(self):
        if not self.similarity_matrix.any():
            messagebox.showerror("Error", "No documents loaded.")
            return

        plt.figure(figsize=(8, 6))
        plt.imshow(self.similarity_matrix, cmap='coolwarm', interpolation='nearest')
        plt.xticks(range(len(self.doc_names)), self.doc_names, rotation=45)
        plt.yticks(range(len(self.doc_names)), self.doc_names)
        plt.colorbar()
        plt.title("Document Similarity Heatmap")
        plt.tight_layout()
        plt.show()

    def show_wordclouds(self):
        for name, text in self.processed_documents.items():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Word Cloud - {name}")
            plt.show()

    def show_top_words(self):
        for name, text in self.processed_documents.items():
            words = word_tokenize(text)
            freq = {}
            for word in words:
                freq[word] = freq.get(word, 0) + 1
            sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
            if sorted_freq:
                words_, counts = zip(*sorted_freq)
                plt.figure(figsize=(8, 4))
                plt.bar(words_, counts, color='skyblue')
                plt.title(f"Top Words in {name}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

    def analyze_sentiment(self):
        sia = SentimentIntensityAnalyzer()
        result = "\nSentiment Scores:\n"
        for name, text in self.documents.items():
            scores = sia.polarity_scores(text)
            result += f"{name}: {scores}\n"
        messagebox.showinfo("Sentiment Analysis", result)

    def detect_plagiarism(self, threshold=0.8):
        result = ""
        for i in range(len(self.doc_names)):
            for j in range(i + 1, len(self.doc_names)):
                sim = self.similarity_matrix[i][j]
                if sim > threshold:
                    result += f"Potential plagiarism: {self.doc_names[i]} <-> {self.doc_names[j]} ({sim:.2f})\n"
        if result:
            messagebox.showwarning("Plagiarism Detected", result)
        else:
            messagebox.showinfo("Result", "No plagiarism detected.")

    def export_csv(self, filename="similarity_report.csv"):
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Document"] + self.doc_names)
                for i, doc in enumerate(self.doc_names):
                    row = [doc] + [f"{self.similarity_matrix[i][j]:.2f}" for j in range(len(self.doc_names))]
                    writer.writerow(row)
            messagebox.showinfo("Exported", f"CSV saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export CSV: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()
