from flask import Flask, render_template, request
from t5_summarizer import summarize_t5
from bart_summarizer import summarize_bart
from pegasus_summarizer import summarize_pegasus

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        article = request.form["article"]
        model = request.form["model"]
        variant = request.form.get("variant")  # For T5 only

        if model == "t5":
            summary = summarize_t5(article, variant)
        elif model == "bart":
            summary = summarize_bart(article)
        elif model == "pegasus":
            summary = summarize_pegasus(article)

    return render_template("index.html", summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
