import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class TFIDF():
    def __init__(self, df, text_col, topic_col):
        super(TFIDF, self).__init__()
        self.df = df.copy()
        self.topic = topic_col
        self.text = text_col
        
        self.tfidf, self.count, self.dft = self.c_tf_idf(self.df, m=len(self.df))
        # self.topn = self.extract_top_n_words_per_topic(n=topk)

    def cosine_similarity(self,a,b):
        num = a @ b.T
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return num/denom

    def c_tf_idf(self, df, m, ngram_range=(1, 1)):
        dft = df.groupby([self.topic], as_index = False).agg({self.text: ' '.join})

        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(dft.body.values)
        t = count.transform(dft.body.values).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count, dft

    def extract_top_n_words_per_topic(self, n=20):
        words = self.count.get_feature_names_out()
        labels = list(self.dft[self.topic])
        tf_idf_transposed = self.tfidf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words

    def reduce_clusters(self, epochs=20):
        for i in range(epochs):
            # Calculate cosine similarity
            similarities = self.cosine_similarity(self.tfidf,self.tfidf)
            np.fill_diagonal(similarities, 0)

            # Extract label to merge into and from where
            topic_sizes = self.df.groupby([self.topic]).count().sort_values(self.text, ascending=False).reset_index()
            topic_to_merge = int(topic_sizes.iloc[-1].topic)
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

            # Adjust topics
            self.df.loc[self.df.topic == topic_to_merge, self.topic] = topic_to_merge_into
            old_topics = self.df.sort_values(self.topic).topic.unique()
            map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}

            self.df['topic_'+str(i)] = self.df.topic.values
            self.df.topic = self.df.topic.map(map_topics)

            # Calculate new topic words
            m = len(self.df)
            self.tfidf, self.count, self.dft = self.c_tf_idf(self.df, m)