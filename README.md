# Unreally API
Detecting fake news on social media

![image05](https://user-images.githubusercontent.com/64489325/189522813-8cd8f4d9-8d00-4bbb-9cda-62a943836dc1.jpg)

## Start locally 🚀
clone this repo
```
git clone https://github.com/unreally-ai/api-twitterbot.git
```

Install requirements
```
cd api-twitterbot/
pip install -r requirements.txt
```

### Executables
⚠️Disclaimer: You can only use these if you have the keys.py file in the api-twitterbot file.
#### API
```
python3 api.py
```
If you now go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) , you can test the API with a GUI.

Here is an example:

#### Twitter Bot
```
pyhton3 twitter_bot.py
```
Now you can mention @calctruth under a tweet, and it will evaluate!
(Make sure you're not running the bot for too long ;))
