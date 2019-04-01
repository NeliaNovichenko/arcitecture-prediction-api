from app import app
from flask import Flask, session
# from flask.ext.session import Session

# sess = Session()

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    # sess.init_app(app)
    app.run()