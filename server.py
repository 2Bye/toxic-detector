# +
import asyncio
import json
import websockets
import argparse
from detox import FilterModel

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--address', type=str, default='0.0.0.0',
                    help='Address to bind service')
parser.add_argument('--port', type=int, default = 6612,
                    help='Port to bind service')
args = parser.parse_args()

detox = FilterModel()

async def detox_service(websocket, path):
    async for data in websocket:
        try:
            pack = json.loads(data)
            user_message = pack['user_message']
            result = detox.get_label(user_message)
            data = {'event' : 'success', 'result' : result}
            await websocket.send(json.dumps(data))
            
        except Exception as e:
            print(f"Error: {e}")
            data = {'event': 'error', 'msg': f'Ошибка запуска! {e}'}
            await websocket.send(json.dumps(data))
            
print('WS Server started.\n')
asyncio.get_event_loop().run_until_complete( 
websockets.serve(detox_service, args.address, args.port, max_size=1024*1024*10))
asyncio.get_event_loop().run_forever()    
