import redis
import json

def test_redis_connection():
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test connection
        response = r.ping()
        print(f"✅ Redis connection successful: {response}")
        
        # Test basic operations
        r.set('test_key', 'Hello Redis!')
        value = r.get('test_key')
        print(f"✅ Redis test value: {value}")
        
        # Test JSON storage (for your cache)
        test_data = {'embeddings': [1, 2, 3, 4, 5], 'timestamp': '2024-01-01'}
        r.set('test_json', json.dumps(test_data))
        retrieved_data = json.loads(r.get('test_json'))
        print(f"✅ Redis JSON test: {retrieved_data}")
        
        # Clean up test keys
        r.delete('test_key', 'test_json')
        
        return True
        
    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

if __name__ == "__main__":
    test_redis_connection()