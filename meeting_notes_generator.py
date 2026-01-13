import os
import sys
import time
import subprocess
import threading
import signal
import re
from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

# Import your existing generator
try:
    from meeting_note_generator_claude import AdaptiveMeetingNotesGenerator
except ImportError:
    print("❌ Error: Could not find meeting_note_generator_claude.py in the same directory.")
    sys.exit(1)

class RenaMeetingBot:
    """
    Rena AI Meeting Bot v1.1
    Optimized for stable audio recording and auto-configuration.
    """
    
    def __init__(self, bot_name="Rena AI (Note Taker)"):
        self.bot_name = bot_name
        self.output_dir = Path("meeting_outputs") / "recordings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = False
        self.audio_process = None
        self.recording_path = None

    def get_audio_device_name(self, ffmpeg_exe):
        """Finds the actual name of the VB-CABLE device on this system."""
        try:
            cmd = [ffmpeg_exe, "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            output = res.stderr
            matches = re.findall(r'"(CABLE Output [^"]+)"', output)
            if matches:
                return f"audio={matches[0]}"
            return "audio=CABLE Output (VB-Audio Virtual Cable)"
        except:
            return "audio=CABLE Output (VB-Audio Virtual Cable)"

    def start_audio_recording(self, filename):
        """Starts FFmpeg with auto-device detection and signal boost."""
        self.recording_path = self.output_dir / f"{filename}.wav"
        
        # Discover FFmpeg
        ffmpeg_exe = "ffmpeg"
        common_paths = [
            r"C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
        ]
        for p in common_paths:
            if os.path.exists(p):
                ffmpeg_exe = p
                break
        
        device_name = self.get_audio_device_name(ffmpeg_exe)
        print(f"🎬 Using FFmpeg at: {ffmpeg_exe}")
        print(f"🎙️ Target Device: {device_name}")
        print(f"🎙️ Starting recording: {self.recording_path}")

        # Command with low-latency and slight volume boost (2x) to fix silent/low recordings
        command = [
            ffmpeg_exe, '-y',
            '-f', 'dshow',
            '-i', device_name,
            '-af', 'volume=2.0',
            str(self.recording_path)
        ]
        
        try:
            self.audio_process = subprocess.Popen(
                command, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            self.is_recording = True
        except Exception as e:
            print(f"❌ FAILED TO START RECORDING: {e}")
            self.is_recording = False

    def stop_audio_recording(self):
        """Gracefully stops the FFmpeg process."""
        if self.audio_process:
            print("🛑 Stopping recording...")
            self.audio_process.send_signal(signal.CTRL_BREAK_EVENT)
            self.audio_process.wait()
            self.is_recording = False
            print(f"✅ Recording saved to: {self.recording_path}")

    def join_google_meet(self, meet_url):
        """Launches browser and handles the joining flow."""
        with sync_playwright() as p:
            user_data_dir = os.path.join(os.getcwd(), "bot_session")
            
            context = p.chromium.launch_persistent_context(
                user_data_dir,
                headless=False,
                viewport={'width': 1280, 'height': 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                args=[
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                    "--disable-blink-features=AutomationControlled"
                ]
            )
            
            page = context.pages[0] if context.pages else context.new_page()
            stealth = Stealth()
            stealth.apply_stealth_sync(page)
            
            print(f"🚀 Navigating to: {meet_url}")
            page.goto(meet_url)
            
            try:
                print("⏳ Waiting for page to load (10s)...")
                time.sleep(10) 
                
                # Check for blocking or dialogs
                try: page.click("text=Dismiss", timeout=3000)
                except: pass

                # Input Name
                try:
                    name_input = page.wait_for_selector('input[aria-label="Your name"], input[type="text"]', timeout=5000)
                    if name_input:
                        name_input.click()
                        name_input.fill(self.bot_name)
                        page.keyboard.press("Enter")
                except: pass
                
                # Turn off Cam/Mic
                time.sleep(2)
                page.keyboard.press("Control+e")
                page.keyboard.press("Control+d")
                print("🔇 Camera and Mic toggled off.")

                # Click Join
                join_selectors = [
                    'span:has-text("Ask to join")', 'span:has-text("Join now")',
                    'button:has-text("Ask to join")', 'button:has-text("Join now")'
                ]
                for selector in join_selectors:
                    if page.is_visible(selector):
                        page.locator(selector).first.click()
                        print("📩 Join request sent.")
                        break

                # Wait for admission
                print("⏳ Waiting for admission...")
                page.wait_for_selector('button[aria-label="Leave call"]', timeout=300000) 
                print("🎉 Bot is in the meeting.")

                # --- AUTO-CONFIGURE SPEAKER ---
                try:
                    print("⚙️ Auto-configuring audio output to VB-CABLE...")
                    page.click('button[aria-label="More options"]')
                    page.click('text="Settings"')
                    time.sleep(1)
                    # Use a broad selector for speakers dropdown
                    page.click('[aria-label="Speakers"]')
                    time.sleep(0.5)
                    # Click the virtual cable option
                    page.click('text="CABLE Input (VB-Audio Virtual Cable)"')
                    page.keyboard.press("Escape")
                    print("✅ Speaker successfully set to Virtual Cable.")
                except:
                    print("⚠️ Could not auto-set speaker. Please manually set Speaker to 'CABLE Input' in Google Meet settings.")

                # Start Recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.start_audio_recording(f"meeting_{timestamp}")

                # Monitor
                while True:
                    if not page.is_visible('button[aria-label="Leave call"]'):
                        print("📴 Meeting ended.")
                        break
                    time.sleep(5)

            except Exception as e:
                print(f"⚠️ Session interrupted: {e}")
            finally:
                self.stop_audio_recording()
                try: context.close()
                except: pass
                
                if self.recording_path and os.path.exists(self.recording_path):
                    self.run_ai_pipeline()

    def run_ai_pipeline(self):
        """Triggers the analysis pipeline."""
        print("\n" + "="*60)
        print("🤖 STARTING AI ANALYSIS")
        print("="*60)
        generator = AdaptiveMeetingNotesGenerator()
        generator.process_meeting(str(self.recording_path))
        print("🎉 COMPLETE!")

if __name__ == "__main__":
    try:
        url = sys.argv[1] if len(sys.argv) > 1 else ""
        if not url:
            print("Usage: python rena_bot_pilot.py <MEET_URL>")
            sys.exit(1)
        RenaMeetingBot().join_google_meet(url)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
