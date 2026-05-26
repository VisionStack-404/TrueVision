from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import uvicorn
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# SMTP Configuration (Gmail)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "varunvadlakonda1577@gmail.com")
SMTP_PASS = os.getenv("SMTP_PASS", "gqzdtofplqrwqvhw")
SMTP_SENDER_NAME = "TrueVision Team"

app = FastAPI(
    title="TrueVision Proxy API",
    description="Proxy server for Deepfake Detection on EC2",
    version="1.0.0"
)

# EC2 Endpoints
EC2_PROCESS_URL = "http://3.238.89.41:8000/process/"
EC2_CAPABILITIES_URL = "http://3.238.89.41:8000/detection/capabilities/"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Root Endpoint
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "TrueVision Proxy Server is Live",
        "available_endpoints": {
            "process": "/process/",
            "capabilities": "/detection/capabilities/",
            "docs": "/docs"
        }
    }


# ✅ GET: Fetch Detection Capabilities from EC2
@app.get("/detection/capabilities/")
def get_capabilities():
    try:
        response = requests.get(EC2_CAPABILITIES_URL, timeout=10)
        response.raise_for_status()
        return JSONResponse(content=response.json())
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch capabilities from EC2: {str(e)}"
        )


# ✅ POST: Forward File to EC2 for Processing
@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        files = {
            "file": (
                file.filename,
                file_bytes,
                file.content_type or "application/octet-stream"
            )
        }

        response = requests.post(
            EC2_PROCESS_URL,
            files=files,
            timeout=600
        )
        response.raise_for_status()

        return JSONResponse(content=response.json())

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="EC2 request timed out.")

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to the EC2 server."
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Request error: {str(e)}"
        )


# ✅ Informational GET Endpoint for /process/
@app.get("/process/")
def process_info():
    return {
        "message": "Use POST to upload an image or video for deepfake detection.",
        "required_field": "file",
        "supported_formats_endpoint": "/detection/capabilities/",
        "example_curl": (
            "curl -X POST 'http://127.0.0.1:8000/process/' "
            "-H 'accept: application/json' "
            "-F 'file=@sample.jpg'"
        )
    }

# ✅ POST: Forward feedback label to EC2 for online learning
# Expected payload: {"label": "FAKE"} or {"label": "REAL"}
@app.post("/feedback/")
async def forward_feedback(payload: dict):
    try:
        # Forward the exact payload to EC2 feedback endpoint
        response = requests.post(
            "http://3.238.89.41:8000/feedback/",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        # Return whatever EC2 sends back (usually a status/message)
        return JSONResponse(content=response.json())
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to forward feedback to EC2: {str(e)}",
        )



# ── Email payload model ─────────────────────────────────────────
class WelcomeEmailPayload(BaseModel):
    name: str
    email: str
    is_new_user: bool = True   # True = signup, False = login


# ✅ POST: Send welcome / login notification email via AWS SES
@app.post("/send-welcome-email/")
async def send_welcome_email(payload: WelcomeEmailPayload):
    first_name = payload.name.split()[0]
    subject = (
        f"Welcome to TrueVision — from TrueVision Team! 🛡️"
        if payload.is_new_user
        else f"Welcome back to TrueVision — from TrueVision Team! 👋"
    )

    html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{subject}</title>
</head>
<body style="margin:0;padding:0;background:#060610;font-family:'Inter',Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#060610;padding:40px 16px;">
  <tr><td align="center">
    <table width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%;">

      <!-- Header -->
      <tr><td style="background:linear-gradient(135deg,#1a0a3a,#0d0d1f);border-radius:20px 20px 0 0;padding:40px 40px 32px;text-align:center;border:1px solid rgba(124,58,237,.25);border-bottom:none;">
        <div style="width:64px;height:64px;background:linear-gradient(135deg,#7c3aed,#9333ea);border-radius:18px;display:inline-flex;align-items:center;justify-content:center;margin-bottom:20px;box-shadow:0 0 30px rgba(124,58,237,.4);">
          <span style="color:#fff;font-size:28px;">🛡️</span>
        </div>
        <h1 style="color:#fff;font-size:26px;font-weight:800;margin:0 0 8px;letter-spacing:-0.5px;">TrueVision</h1>
        <p style="color:rgba(167,139,250,.8);font-size:12px;margin:0;text-transform:uppercase;letter-spacing:2px;">Forensic AI Engine</p>
      </td></tr>

      <!-- Body -->
      <tr><td style="background:rgba(15,10,30,.95);padding:36px 40px;border:1px solid rgba(124,58,237,.2);border-top:none;border-bottom:none;">
        <h2 style="color:#f0f0f8;font-size:22px;font-weight:700;margin:0 0 16px;">
          {"Welcome aboard! 🎉" if payload.is_new_user else "Welcome back! 👋"}
        </h2>
        <p style="color:rgba(255,255,255,.55);font-size:15px;line-height:1.7;margin:0 0 24px;">
          Hi <strong style="color:#a78bfa;">{payload.name}</strong>,<br><br>
          {"We are thrilled to have you join our forensic family. Your TrueVision workspace is now fully configured for high-precision deepfake detection."
           if payload.is_new_user
           else
           "Your forensic workspace has been restored. You can continue analyzing digital assets with the latest neural engine updates."}<br><br>
           Our platform utilizes a <strong>Tri-Model Ensemble Architecture</strong> (CNN, CViT, and ETCNN) trained on millions of samples from FaceForensics++ and DFDC to provide industry-leading 99.2% accuracy.
        </p>

        <!-- Feature pills -->
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:28px;">
          <tr>
            <td style="padding:4px;">
              <div style="background:rgba(124,58,237,.12);border:1px solid rgba(124,58,237,.25);border-radius:12px;padding:14px 16px;text-align:center;">
                <div style="font-size:22px;margin-bottom:6px;">🧠</div>
                <div style="color:#c4b5fd;font-size:12px;font-weight:600;">CNN Model</div>
                <div style="color:rgba(255,255,255,.3);font-size:11px;">Face-swap detection</div>
              </div>
            </td>
            <td style="padding:4px;">
              <div style="background:rgba(124,58,237,.12);border:1px solid rgba(124,58,237,.25);border-radius:12px;padding:14px 16px;text-align:center;">
                <div style="font-size:22px;margin-bottom:6px;">👁️</div>
                <div style="color:#c4b5fd;font-size:12px;font-weight:600;">CViT Model</div>
                <div style="color:rgba(255,255,255,.3);font-size:11px;">Boundary artifacts</div>
              </div>
            </td>
            <td style="padding:4px;">
              <div style="background:rgba(124,58,237,.12);border:1px solid rgba(124,58,237,.25);border-radius:12px;padding:14px 16px;text-align:center;">
                <div style="font-size:22px;margin-bottom:6px;">⚡</div>
                <div style="color:#c4b5fd;font-size:12px;font-weight:600;">ETCNN Model</div>
                <div style="color:rgba(255,255,255,.3);font-size:11px;">Texture analysis</div>
              </div>
            </td>
          </tr>
        </table>

        <!-- CTA button -->
        <div style="text-align:center;margin-bottom:28px;">
          <a href="http://localhost:5500" style="display:inline-block;background:linear-gradient(135deg,#7c3aed,#9333ea);color:#fff;text-decoration:none;font-weight:700;font-size:15px;padding:14px 36px;border-radius:12px;box-shadow:0 8px 24px rgba(124,58,237,.35);">
            Open Dashboard →
          </a>
        </div>

        <!-- Info strip -->
        <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:16px 20px;">
          <p style="color:rgba(255,255,255,.25);font-size:12px;margin:0;line-height:1.6;">
            📧 <strong style="color:rgba(255,255,255,.4);">Account:</strong> {payload.email}<br>
            🕒 <strong style="color:rgba(255,255,255,.4);">Session time:</strong> Just now<br>
            🔒 Your analysis history is stored locally and never shared.
          </p>
        </div>
      </td></tr>

      <!-- Footer -->
      <tr><td style="background:rgba(6,6,16,.8);border:1px solid rgba(124,58,237,.15);border-top:1px solid rgba(255,255,255,.04);border-radius:0 0 20px 20px;padding:24px 40px;text-align:center;">
        <p style="color:rgba(255,255,255,.15);font-size:12px;margin:0;line-height:1.7;">
          TrueVision Forensic AI · Deepfake Detection Platform<br>
          <span style="color:rgba(255,255,255,.3);">Best Regards, <br> <b>The TrueVision Team</b></span><br><br>
          <span style="color:rgba(255,255,255,.08);">
            You're receiving this because you {'signed up for' if payload.is_new_user else 'logged into'} TrueVision.
          </span>
        </p>
      </td></tr>

    </table>
  </td></tr>
</table>
</body>
</html>
"""

    text_body = (
        f"Welcome to TrueVision, {payload.name}!\n\n"
        f"Your forensic AI workspace is now ready.\n"
        f"Dashboard: http://localhost:5500\n\n"
        f"Models: CNN · CViT · ETCNN ensemble\n"
        f"Account: {payload.email}"
    )

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{SMTP_SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = payload.email

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, payload.email, msg.as_string())

        return {"status": "success", "message": f"Welcome email from TrueVision Team sent to {payload.email}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SMTP error: {str(e)}")

# ── Report Email payload model ─────────────────────────────────────────
class ReportEmailPayload(BaseModel):
    name: str
    email: str
    file_name: str
    prediction: str
    confidence: float
    explanation: str


# ✅ POST: Send Analysis Report Email via AWS SES
@app.post("/send-report-email/")
async def send_report_email(payload: ReportEmailPayload):
    subject = f"TrueVision Forensic Report: {payload.file_name} [{payload.prediction}]"
    
    is_fake = payload.prediction == "FAKE"
    color_hex = "#ef4444" if is_fake else "#10b981"
    icon = "🚨" if is_fake else "✅"

    html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{subject}</title>
</head>
<body style="margin:0;padding:0;background:#050508;font-family:'Inter',Arial,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#050508;padding:40px 16px;">
  <tr><td align="center">
    <table width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%;">
      <!-- Header -->
      <tr><td style="background:linear-gradient(135deg,#1a0a3a,#0d0d1f);border-radius:20px 20px 0 0;padding:40px 40px 24px;text-align:center;border:1px solid rgba(124,58,237,.25);border-bottom:none;">
        <h1 style="color:#fff;font-size:24px;font-weight:800;margin:0 0 4px;letter-spacing:-0.5px;">TrueVision Report</h1>
        <p style="color:rgba(167,139,250,.8);font-size:12px;margin:0;text-transform:uppercase;letter-spacing:2px;">Requested by {payload.name}</p>
      </td></tr>
      
      <!-- Body -->
      <tr><td style="background:rgba(15,10,30,.95);padding:36px 40px;border:1px solid rgba(124,58,237,.2);border-top:none;border-bottom:none;">
        <div style="text-align:center;margin-bottom:30px;">
          <p style="color:rgba(255,255,255,.5);font-size:14px;margin:0 0 10px;">Analysis for file:</p>
          <div style="background:rgba(255,255,255,.05);padding:10px;border-radius:8px;font-family:monospace;color:#fff;">{payload.file_name}</div>
        </div>

        <!-- Verdict Box -->
        <table width="100%" cellpadding="0" cellspacing="0" style="background:rgba(255,255,255,.02);border:1px solid {color_hex}40;border-radius:16px;margin-bottom:24px;">
          <tr><td style="padding:24px;text-align:center;">
            <p style="margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:rgba(255,255,255,.4);">Final Verdict</p>
            <h2 style="margin:0 0 8px;font-size:36px;font-weight:900;letter-spacing:2px;color:{color_hex};">
              {icon} {payload.prediction}
            </h2>
            <p style="margin:0;font-size:14px;color:rgba(255,255,255,.7);">Confidence: <strong>{payload.confidence}%</strong></p>
          </td></tr>
        </table>

        <!-- Explanation -->
        <h3 style="color:#fff;font-size:16px;font-weight:600;margin:0 0 12px;border-bottom:1px solid rgba(255,255,255,.1);padding-bottom:8px;">Forensic Details</h3>
        <p style="color:rgba(255,255,255,.65);line-height:1.6;font-size:14px;margin:0 0 24px;">
          {payload.explanation}
        </p>
        
        <div style="text-align:center;margin-top:40px;">
          <a href="http://localhost:5500" style="display:inline-block;background:linear-gradient(135deg,#7c3aed,#9333ea);color:#fff;text-decoration:none;font-weight:700;font-size:14px;padding:12px 32px;border-radius:10px;">
            View in Dashboard →
          </a>
        </div>
      </td></tr>

      <!-- Footer -->
      <tr><td style="background:rgba(6,6,16,.8);border:1px solid rgba(124,58,237,.15);border-top:1px solid rgba(255,255,255,.04);border-radius:0 0 20px 20px;padding:24px 40px;text-align:center;">
        <p style="color:rgba(255,255,255,.15);font-size:12px;margin:0;line-height:1.7;">
          Confidential TrueVision Forensic Report.<br>
          <span style="color:rgba(255,255,255,.3);">Best Regards, <br> <b>The TrueVision Team</b></span><br><br>
          Sent securely to {payload.email}
        </p>
      </td></tr>
    </table>
  </td></tr>
</table>
</body>
</html>
"""
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = f"{SMTP_SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = payload.email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, payload.email, msg.as_string())

        return {"status": "success", "message": f"Forensic report emailed to {payload.email}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SMTP error: {str(e)}")

# ── Audit Report payload model ─────────────────────────────────────────
class AuditItem(BaseModel):
    file_name: str
    prediction: str
    confidence: str
    timestamp: str

class AuditEmailPayload(BaseModel):
    name: str
    email: str
    total_real: int
    total_fake: int
    history: list[AuditItem]

@app.post("/send-audit-report/")
async def send_audit_report(payload: AuditEmailPayload):
    subject = f"TrueVision Account Activity Audit — {len(payload.history)} Reports"
    
    rows_html = ""
    for item in payload.history:
        color = "#ef4444" if item.prediction == "FAKE" else "#10b981"
        rows_html += f"""
        <tr>
            <td style="padding:12px; border-bottom:1px solid rgba(255,255,255,.05); color:#fff; font-size:13px;">{item.file_name}</td>
            <td style="padding:12px; border-bottom:1px solid rgba(255,255,255,.05); color:{color}; font-weight:700; font-size:12px;">{item.prediction}</td>
            <td style="padding:12px; border-bottom:1px solid rgba(255,255,255,.05); color:rgba(255,255,255,.6); font-size:12px;">{item.confidence}</td>
            <td style="padding:12px; border-bottom:1px solid rgba(255,255,255,.05); color:rgba(255,255,255,.4); font-size:11px;">{item.timestamp}</td>
        </tr>
        """

    html_body = f"""
<!DOCTYPE html>
<html>
<body style="background:#050508; color:#fff; font-family:sans-serif; padding:40px;">
    <div style="max-width:600px; margin:0 auto; background:#0e0c1a; border:1px solid #1a0a3a; border-radius:16px; overflow:hidden;">
        <div style="background:linear-gradient(135deg,#1a0a3a,#0d0d1f); padding:30px; text-align:center;">
            <h1 style="margin:0; font-size:20px;">Activity Audit Report</h1>
            <p style="margin:5px 0 0; font-size:12px; color:#8b5cf6;">TRUEVISION FORENSIC SYSTEMS</p>
        </div>
        <div style="padding:30px;">
            <p>Hi {payload.name},</p>
            <p style="color:rgba(255,255,255,.6); font-size:14px;">Here is the complete summary of all deepfake detection activities associated with your account.</p>
            
            <div style="display:flex; justify-content:space-between; margin:24px 0; gap:10px;">
                <div style="flex:1; background:rgba(255,255,255,0.03); padding:15px; border-radius:10px; text-align:center; border:1px solid rgba(255,255,255,0.05);">
                    <div style="font-size:20px; font-weight:800; color:#fff;">{len(payload.history)}</div>
                    <div style="font-size:10px; color:rgba(255,255,255,0.4); text-transform:uppercase; margin-top:4px;">Total Scans</div>
                </div>
                <div style="flex:1; background:rgba(16,185,129,0.05); padding:15px; border-radius:10px; text-align:center; border:1px solid rgba(16,185,129,0.1);">
                    <div style="font-size:20px; font-weight:800; color:#10b981;">{payload.total_real}</div>
                    <div style="font-size:10px; color:rgba(16,185,129,0.5); text-transform:uppercase; margin-top:4px;">Real Found</div>
                </div>
                <div style="flex:1; background:rgba(239,68,68,0.05); padding:15px; border-radius:10px; text-align:center; border:1px solid rgba(239,68,68,0.1);">
                    <div style="font-size:20px; font-weight:800; color:#ef4444;">{payload.total_fake}</div>
                    <div style="font-size:10px; color:rgba(239,68,68,0.5); text-transform:uppercase; margin-top:4px;">Fakes Detected</div>
                </div>
            </div>
            <table width="100%" cellspacing="0" cellpadding="0" style="margin-top:20px; border-collapse:collapse;">
                <thead>
                    <tr style="text-align:left; background:rgba(255,255,255,.02);">
                        <th style="padding:12px; font-size:11px; text-transform:uppercase; color:rgba(255,255,255,.4);">File</th>
                        <th style="padding:12px; font-size:11px; text-transform:uppercase; color:rgba(255,255,255,.4);">Verdict</th>
                        <th style="padding:12px; font-size:11px; text-transform:uppercase; color:rgba(255,255,255,.4);">Conf.</th>
                        <th style="padding:12px; font-size:11px; text-transform:uppercase; color:rgba(255,255,255,.4);">Date</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            <div style="margin-top:40px; border-top:1px solid rgba(255,255,255,.1); padding-top:20px; text-align:center;">
               <p style="font-size:11px; color:rgba(255,255,255,.3);">This is an automated security audit generated by TrueVision Team.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = f"{SMTP_SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = payload.email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, payload.email, msg.as_string())

        return {"status": "success", "message": f"Audit report emailed to {payload.email}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SMTP error: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)