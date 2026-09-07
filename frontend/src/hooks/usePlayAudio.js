import { useRef, useState, useCallback } from "react";

export function usePlayAudio() {
  const audioRef = useRef(null);
  const [playingId, setPlayingId] = useState(null);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setPlayingId(null);
  }, []);

  const togglePlay = useCallback(
    (id, audioBlob) => {
      if (playingId === id) {
        stop();
        return;
      }

      stop();

      const url = URL.createObjectURL(audioBlob);
      const audio = new Audio(url);
      audioRef.current = audio;

      audio.onended = () => {
        setPlayingId(null);
        URL.revokeObjectURL(url);
      };

      audio.play();
      setPlayingId(id);
    },
    [playingId, stop]
  );

  return { playingId, togglePlay };
}