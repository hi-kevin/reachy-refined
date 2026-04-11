"""Test: does the live mic audio get recognized with VAD config?

Run on robot:
  cd ~/reachy-refined && source .venv/bin/activate && python scripts/test_live_api.py
"""

import asyncio
import os
import sys
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-live-preview"
API_KEY = os.getenv("GOOGLE_API_KEY")


async def test_with_real_mic():
    """Connect with VAD + transcription, capture real mic audio, see if Gemini responds."""
    print(f"\n=== TEST: Real mic audio with VAD config ===")
    client = genai.Client(api_key=API_KEY)

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=20,
                silence_duration_ms=500,
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[types.Part(text="You are a friendly robot. Be concise.")]
        ),
    )

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("  Connected!")

            # Send greeting text to get initial response
            await session.send_realtime_input(text="Say hello briefly.")
            print("  Sent greeting.")

            # Also try sending mic audio from robot
            from reachy_mini import ReachyMini
            robot = ReachyMini()
            await asyncio.sleep(2)  # Wait for WebRTC
            robot.media.start_recording()
            print("  Mic recording started.")

            async def send_mic():
                for i in range(200):  # ~20 seconds
                    sample = robot.media.get_audio_sample()
                    if sample is not None and len(sample) > 0:
                        # Convert to int16 PCM
                        if sample.dtype == np.float32:
                            if len(sample.shape) == 2 and sample.shape[1] == 2:
                                sample = np.mean(sample, axis=1)
                            audio_int16 = (np.clip(sample, -1.0, 1.0) * 32767).astype(np.int16)
                        else:
                            audio_int16 = sample
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=audio_int16.tobytes(),
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                        if i == 0:
                            print(f"  First mic chunk: {len(audio_int16)} samples, dtype={audio_int16.dtype}")
                    await asyncio.sleep(0.1)

            async def recv_responses():
                count = 0
                async for resp in session.receive():
                    count += 1
                    sc = resp.server_content
                    if sc:
                        if hasattr(sc, 'input_transcription') and sc.input_transcription:
                            print(f"  >>> INPUT TRANSCRIPT: {sc.input_transcription.text}")
                        if hasattr(sc, 'output_transcription') and sc.output_transcription:
                            print(f"  <<< OUTPUT TRANSCRIPT: {sc.output_transcription.text}")
                        if sc.model_turn:
                            for p in sc.model_turn.parts:
                                if p.inline_data:
                                    print(f"  AUDIO: {len(p.inline_data.data)} bytes", end="")
                        if sc.turn_complete:
                            print(f"\n  --- TURN COMPLETE ---")
                        if sc.generation_complete:
                            print(f"\n  --- GENERATION COMPLETE ---")
                    if resp.session_resumption_update:
                        if count <= 3 or count % 20 == 0:
                            idx = resp.session_resumption_update.last_consumed_client_message_index
                            print(f"  RESUMPTION #{count}: consumed={idx}")
                    if resp.voice_activity:
                        print(f"  VAD: {resp.voice_activity}")
                    if count > 300:
                        break

            print("  Speak into the robot mic now! (20 seconds)")
            try:
                async with asyncio.timeout(25):
                    await asyncio.gather(send_mic(), recv_responses())
            except TimeoutError:
                print("  Timeout reached.")
            finally:
                robot.media.stop_recording()
                robot.disconnect()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print(f"Model: {MODEL}")
    await test_with_real_mic()
    print("\n=== DONE ===")


if __name__ == "__main__":
    asyncio.run(main())
