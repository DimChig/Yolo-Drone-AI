import os
import subprocess

from dotenv.variables import Variable
from moviepy.video.fx.all import speedx as speedx_video
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip
from pydub import AudioSegment


def cut_video(input_path, timecode_start, timecode_end, output_path):
    start_time = timecode_start
    duration = timecode_end - timecode_start  # 1 minute

    # Modify ffmpeg command to place -ss after -i for more accurate seeking
    # Use '-c copy' to avoid re-encoding
    command = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', input_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',  # Re-encode video
        '-c:a', 'aac',  # Copy audio or re-encode if necessary
        output_path
    ]
    subprocess.run(command, check=True)


def trim_and_speed_up_video(video_path, new_speed, new_duration, output_path):
    """
    Trims a video to the specified new duration and speeds it up by the given factor.
    Extracts audio from the video, speeds it up separately, then applies it back to the video.
    Saves the result to the output path.

    :param video_path: The path to the input video.
    :param new_speed: The speed factor to apply (e.g., 1.5 will make the video 50% faster).
    :param new_duration: The desired duration of the trimmed video (in seconds).
    :param output_path: The path where the output video should be saved (including filename).
    """
    temp_audio_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
    sped_up_audio_path = os.path.join(os.path.dirname(video_path), "temp_sped_up_audio.wav")
    temp_video_path = os.path.join(os.path.dirname(video_path), "temp_video.mp4")

    try:
        # Load the video file
        video = VideoFileClip(video_path)

        # Extract the audio from the video and save it temporarily
        video.audio.write_audiofile(temp_audio_path, logger=None)

        # Process the audio to speed it up
        sound = AudioSegment.from_file(temp_audio_path)
        sped_up_audio = sound.speedup(playback_speed=new_speed)
        sped_up_audio.export(sped_up_audio_path, format="wav")

        # Load the sped-up audio
        new_audio = AudioFileClip(sped_up_audio_path)

        # Speed up the video
        sped_up_video = speedx_video(video, new_speed)

        # Set the modified audio to the sped-up video
        sped_up_video = sped_up_video.set_audio(new_audio)

        # Write the output video
        sped_up_video.write_videofile(temp_video_path, codec="libx264", audio_codec="aac", threads=8, logger=None)

        # Close resources
        video.close()
        new_audio.close()

        # Trim video
        cut_video(temp_video_path, 0, new_duration, output_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    finally:
        # Remove temporary audio files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(sped_up_audio_path):
            os.remove(sped_up_audio_path)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# if __name__ == "__main__":
#     video_path = "C:/Users/DimChig/PycharmProjects/AutoReelGenerator/storage/South-Park-S07E02/clips/clip2/footage_rendered.mp4"
#     output_path = "C:/Users/DimChig/PycharmProjects/AutoReelGenerator/storage/South-Park-S07E02/clips/clip2/footage_rendered_trimmed.mp4"
#
#     trim_and_speed_up_video(video_path, Variables.TRIMMED_VIDEO_SPEED, Variables.TRIMMED_VIDEO_DURATION, output_path)
    #
    # original_clip = mp.VideoFileClip(video_path)
    # audio_path = "C:/Users/DimChig/PycharmProjects/AutoReelGenerator/storage/GravityFalls_S1E1/clips/clip1/audio_original.wav"
    # audio_path2 = "C:/Users/DimChig/PycharmProjects/AutoReelGenerator/storage/GravityFalls_S1E1/clips/clip1/audio_new.wav"

