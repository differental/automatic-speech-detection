async def audio_processor(websocket, path):
    global audio_buffer, accumulated_audio, last_confidence, processed_time_ms, total_length_ms, full_accumulated_audio
    try:
        async for packet in websocket:
            audio_int16 = np.frombuffer(packet, np.int16)
            audio_float32 = int2float(audio_int16)
            #process python

    except exceptions.ConnectionClosed:
        print("Connection closed")

async def main():
    async with websockets.serve(audio_processor, "localhost", 8090):
        await asyncio.Future()

if _name_ == "_main_":
    asyncio.run(main())