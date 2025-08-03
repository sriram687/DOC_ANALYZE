# 🔗 Webhook Integration Guide

## 📡 Your Webhook URL

```
https://doc-analyze.onrender.com/webhook
```

## 🚀 What is a Webhook?

A webhook is an HTTP endpoint that external services can POST data to when events occur. Your API now has a webhook endpoint that can receive notifications from:

- **Payment processors** (Stripe, PayPal)
- **Version control** (GitHub, GitLab)
- **Communication tools** (Slack, Discord)
- **Custom integrations**
- **Third-party services**

## 📋 Webhook Endpoint Details

### **POST /webhook**
- **URL**: `https://doc-analyze.onrender.com/webhook`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Authentication**: None (can be added if needed)

### **Response Format**
```json
{
  "status": "success",
  "message": "Webhook received and processed",
  "webhook_type": "payment.success",
  "timestamp": "2025-08-02T15:30:00.000Z"
}
```

## 🧪 Testing Your Webhook

### **Method 1: Using curl**
```bash
curl -X POST https://doc-analyze.onrender.com/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "type": "test",
    "message": "Hello webhook!",
    "timestamp": "2025-08-02T15:30:00.000Z"
  }'
```

### **Method 2: Using the Test Script**
```bash
cd scripts
python test_webhook.py
```

### **Method 3: Webhook Info Endpoint**
Visit: `https://doc-analyze.onrender.com/webhook/test`

## 🔧 Common Use Cases

### **1. Payment Notifications (Stripe)**
```json
{
  "type": "payment.success",
  "payment_id": "pay_123456",
  "amount": 29.99,
  "currency": "USD",
  "customer_email": "user@example.com"
}
```

### **2. Document Processing**
```json
{
  "type": "document.uploaded",
  "document_id": "doc_789",
  "filename": "contract.pdf",
  "size": 1024000,
  "user_id": "user_456"
}
```

### **3. GitHub Events**
```json
{
  "type": "push",
  "repository": "user/repo",
  "branch": "main",
  "commits": 3
}
```

## 🛠️ Customizing Webhook Processing

The webhook endpoint in `backend/main.py` can be customized to:

1. **Add Authentication**: Verify webhook signatures
2. **Process Specific Events**: Handle different webhook types
3. **Store Data**: Save webhook data to database
4. **Trigger Actions**: Start document processing, send emails, etc.

### **Example: Adding Signature Verification**
```python
import hmac
import hashlib

@app.post("/webhook")
async def webhook_listener(request: Request):
    # Verify webhook signature (example for Stripe)
    signature = request.headers.get("stripe-signature")
    payload = await request.body()
    
    # Verify signature
    if not verify_signature(payload, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Process webhook...
```

## 📊 Monitoring Webhooks

The webhook endpoint logs all incoming requests with:
- **Source IP**: Where the webhook came from
- **Payload**: The data received
- **Headers**: Request headers for debugging
- **Processing Status**: Success or error

Check your Render logs to monitor webhook activity.

## 🔒 Security Best Practices

1. **Verify Signatures**: Always verify webhook signatures from trusted sources
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **IP Whitelisting**: Restrict webhooks to known IP ranges
4. **HTTPS Only**: Always use HTTPS for webhook URLs
5. **Validate Payloads**: Validate incoming data structure

## 🚨 Troubleshooting

### **Webhook Not Receiving Data**
- Check the webhook URL is correct
- Verify the service is sending to the right endpoint
- Check Render logs for errors

### **400 Bad Request**
- Ensure JSON payload is valid
- Check Content-Type header is `application/json`
- Verify payload structure

### **Timeout Issues**
- Webhook processing should be fast (<30 seconds)
- For long operations, use background tasks

## 📚 Next Steps

1. **Test the webhook** using the provided scripts
2. **Configure external services** to send webhooks to your URL
3. **Customize processing logic** for your specific use case
4. **Add authentication** if needed
5. **Monitor webhook activity** in your logs

Your webhook is now ready to receive data from external services! 🎉
