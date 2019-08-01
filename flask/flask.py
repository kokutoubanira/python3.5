from flask import Flask
#flaskを定義
app = Flask(__name__)
#ルートのアドレスに以下を配置
@app.route('/')
def hello():
    name = "Hello World"
    return name

#メイン
if __name__ == "__main__":
    app.run(debug=True)
