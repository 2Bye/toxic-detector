# +
import json
from websocket import create_connection

uri = "ws://0.0.0.0:6612/"
ws = create_connection(uri, ping_interval=None)

user_message = 'Hello!'
data = {'user_message' : user_message}
ws.send(json.dumps(data)) ### send data
responce = json.loads(ws.recv())
print(responce)
ws.close()
