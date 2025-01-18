css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');

body {
    background-color: #f0f2f6;
    font-family: 'Roboto', sans-serif;
}
.chat-message {
    padding: 1.5rem;
    border-radius: 0.8rem;
    margin-bottom: 1rem;
    display: flex;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    transition: all 0.3s ease-in-out;
    opacity: 0;
    transform: translateY(20px);
    animation: fadeIn 0.5s ease-out forwards;
}
@keyframes fadeIn {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 15%;
}
.chat-message .avatar img {
    max-width: 60px;
    max-height: 60px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #fff;
}
.chat-message .message {
    width: 85%;
    padding: 0 1.5rem;
    color: #fff;
    font-size: 1rem;
    line-height: 1.5;
}
.stTextInput > div > div > input {
    background-color: #f0f2f6;
    color: #2b313e;
    border-radius: 0.5rem;
    border: 1px solid #ccc;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    transition: all 0.3s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #4CAF50;
    box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
}
.stButton > button {
    border-radius: 0.5rem;
    font-weight: 500;
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #45a049;
}
.stTextArea > div > div > textarea {
    background-color: #f0f2f6;
    color: #2b313e;
    border-radius: 0.5rem;
    border: 1px solid #ccc;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    transition: all 0.3s ease;
}
.stTextArea > div > div > textarea:focus {
    border-color: #4CAF50;
    box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2017/03/31/23/11/robot-2192617_1280.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2017/11/10/05/48/user-2935527_1280.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

loading_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2017/03/31/23/11/robot-2192617_1280.png">
    </div>
    <div class="message">🤖 Tôi đang đọc tài liệu...</div>
</div>
'''