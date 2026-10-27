"""Offline regression checks: no model downloads, servers, or credentials."""
import ast
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "app" / "models"


def functions(filename, names, scope):
    tree = ast.parse((MODELS / filename).read_text())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), filename, "exec"), scope)
    return scope


class NoiseRegressions(unittest.TestCase):
    def test_segmentation_covers_audio_exactly_once(self):
        vectors = iter([np.array([1., 0.]), np.array([1., 0.]),
                        np.array([0., 1.]), np.array([0., 1.])])
        scope = functions("segmenter.py", {"segment_turn", "_cosine_sim", "_single_segment", "_quietest_point"}, {
            "SPLIT_SEARCH_RADIUS_S": .4,
            "np": np, "time": __import__("time"), "logger": logging.getLogger("test"),
            "FAST_PATH_MAX_DURATION_S": 2.5, "SEG_WIN_S": 2.5, "SPK_CHANGE_SIM": .45, "SUBSEG_MIN_S": 1.,
            "_speech_spans": lambda wav: [(0., 6.25)],
            "_windows_for_span": lambda a, b: [(0., 2.5), (1.25, 3.75), (2.5, 5.), (3.75, 6.25)],
        })
        fake_models = types.ModuleType("app.models")
        fake_models.speaker_diart = types.SimpleNamespace(_embed=lambda *args: next(vectors))
        fake_app = types.ModuleType("app")
        fake_app.models = fake_models
        wav = np.arange(100000, dtype=np.float32)
        with patch.dict(sys.modules, {"app": fake_app, "app.models": fake_models}):
            parts = scope["segment_turn"](wav, 16000)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["end"], parts[1]["start"])
        np.testing.assert_array_equal(np.concatenate([p["wav"] for p in parts]), wav)

    def test_language_gate_on_real_log_cases(self):
        scope = functions("stt.py", {"_language_gate", "_detect_language_restricted"}, {
            "np": np, "_trim_leading_quiet": lambda a: a,
            "MIN_PAIR_MASS": .45, "MIN_PAIR_MASS_SHORT": .75, "SHORT_TURN_STRICT_S": 3.,
            "FOREIGN_RIVAL_MIN_PROB": .15, "FOREIGN_RIVAL_RATIO": .4, "CONTINUATION_MIN_PAIR_MASS": .3,
            "SHORT_DOMINANT_MIN_PAIR_MASS": .6, "SHORT_DOMINANCE_RATIO": 5.,
            "LANG_FAMILY": {"ru": {"uk", "be", "bg", "sr", "hr", "bs", "mk", "sl", "pl", "cs", "sk", "lv", "lt"}},
        })
        gate, pair = scope["_language_gate"], ["en", "ru"]

        class FeatureExtractor:
            nb_max_frames = 3000

            def __call__(self, audio):
                return np.zeros((80, 10))

        def detect(probs):
            model = types.SimpleNamespace(
                feature_extractor=FeatureExtractor(), encode=lambda f: f,
                model=types.SimpleNamespace(detect_language=lambda e: [[(f"<|{l}|>", p) for l, p in probs]]),
            )
            return scope["_detect_language_restricted"](model, np.zeros(16000), pair)

        def check(probs, duration, expected_reason, expected_lang=None, continuation=None):
            lang, _, diag = detect(probs)
            self.assertEqual(gate(diag, pair, duration, continuation), expected_reason)
            if expected_lang:
                self.assertEqual(lang, expected_lang)

        check([("en", .523), ("ko", .265), ("tr", .046), ("ar", .027), ("ja", .025)], 3.84, "foreign_rival")
        check([("en", .501), ("ru", .336), ("de", .016), ("es", .015), ("fr", .014)], 1.79, None, "en")
        check([("ru", .448), ("uk", .278), ("en", .124), ("pl", .017), ("bg", .011)], 6.27, None, "ru")
        # aksentli ruscha "Здравствуйте, как у вас дела?" — latishcha/xorvatcha deb taxmin qilingan
        check([("lv", .591), ("hr", .135), ("ru", .048), ("uk", .035), ("sr", .032)], 3.46, None, "ru")
        check([("lv", .340), ("ru", .273), ("en", .116), ("uk", .052), ("pl", .038)], 3.46, None, "ru")
        # aniq inglizcha qisqa "Hello, how are you doing?"
        check([("en", .671), ("de", .036), ("sv", .033), ("is", .028), ("ru", .027)], 2.82, None, "en")
        # noaniq qisqa bo'lak (en/es teng) — rad etiladi
        check([("en", .446), ("es", .375), ("fr", .050), ("pt", .032), ("it", .024)], 2.69, "low_pair_mass_short")
        check([("is", .886), ("en", .061), ("de", .006), ("es", .004), ("sv", .003)], 3.07, "low_pair_mass")

    def test_language_id_input_drops_leading_silence_only(self):
        trim = functions("stt.py", {"_trim_leading_quiet"}, {
            "np": np, "SAMPLE_RATE": 16000, "LID_TRIM_FRAME_S": .02,
            "LID_TRIM_REL": .1, "LID_TRIM_FLOOR_RMS": .005,
        })["_trim_leading_quiet"]
        speech = np.sin(np.arange(16000) / 5.).astype(np.float32) * .3
        lead = np.random.default_rng(0).standard_normal(9600).astype(np.float32) * .001
        out = trim(np.concatenate([lead, speech, np.zeros(19200, np.float32)]))
        self.assertEqual(out.size, speech.size + 19200)
        np.testing.assert_array_equal(trim(np.zeros(100, np.float32)), np.zeros(100, np.float32))

    def test_short_turn_detects_two_voices_without_losing_samples(self):
        vectors = iter([np.array([1., 0.]), np.array([0., 1.])])
        scope = functions("segmenter.py", {"segment_turn", "_cosine_sim", "_single_segment", "_quietest_point"}, {
            "SPLIT_SEARCH_RADIUS_S": .4,
            "np": np, "time": __import__("time"), "logger": logging.getLogger("test"),
            "FAST_PATH_MAX_DURATION_S": 2.5, "SEG_WIN_S": 2.5,
            "SPK_CHANGE_SIM": .45, "SUBSEG_MIN_S": 1.,
            "_speech_spans": lambda wav: [(0., 3.584)],
        })
        fake_models = types.ModuleType("app.models")
        fake_models.speaker_diart = types.SimpleNamespace(_embed=lambda *args: next(vectors))
        fake_app = types.ModuleType("app")
        fake_app.models = fake_models
        wav = np.arange(57344, dtype=np.float32)
        with patch.dict(sys.modules, {"app": fake_app, "app.models": fake_models}):
            parts = scope["segment_turn"](wav, 16000)
        self.assertEqual(len(parts), 2)
        np.testing.assert_array_equal(np.concatenate([p["wav"] for p in parts]), wav)

    def whisper_scope(self, duration=1.92, mass=.891, prob=.995):
        segment = types.SimpleNamespace(text="Actual speech", start=0., end=duration,
                                        no_speech_prob=.01, avg_logprob=-.2, compression_ratio=1.)
        scope = dict(np=np, time=__import__("time"), logger=logging.getLogger("test"),
                     get_model=lambda: object(), SAMPLE_RATE=16000,
                     _decode_waveform=lambda path: np.ones(int(duration * 16000)),
                     RMS_SILENCE_THRESHOLD=.001, USE_VAD_FILTER=False, _ONNX_AVAILABLE=False,
                     SHORT_TURN_STRICT_S=3., MIN_PAIR_MASS_SHORT=.8, MIN_PAIR_MASS=.65,
                     MIN_LANG_PROB=.75, WHISPER_INITIAL_PROMPT_ENABLED=False,
                     FOREIGN_RIVAL_MIN_PROB=.15, FOREIGN_RIVAL_RATIO=.4, CONTINUATION_MIN_PAIR_MASS=.3,
                     AMBIGUOUS_LANG_PROB=0., SHORT_DOMINANT_MIN_PAIR_MASS=.6, SHORT_DOMINANCE_RATIO=5.,
                     SEG_LANG_CHECK_BELOW=0., SEGMENT_REL_RMS=.3,
                     NO_SPEECH_PROB_THRESHOLD=.6, AVG_LOGPROB_THRESHOLD=-1.,
                     COMPRESSION_RATIO_THRESHOLD=2.4, DEBUG_STT=False,
                     _is_codeswitch_candidate=lambda *args: False,
                     _known_speaker_language_override=lambda *args: False,
                     _session_bootstrap_override=lambda *args: False,
                     _is_phantom_drop=lambda *args: False,
                     _is_low_conf_short_drop=lambda *args: False,
                     _run_transcribe=Mock(return_value=([segment], object())),
                     _detect_language_restricted=lambda *args: ("en", prob, dict(
                         pair_mass=mass, raw_top_lang="en", raw_top_prob=mass*prob,
                         top5=[("en", mass*prob)])))
        functions("stt.py", {"_transcribe_audio_whisper", "_language_gate"}, scope)
        return scope, segment

    def test_short_confident_language_survives_without_known_speaker(self):
        scope, _ = self.whisper_scope()
        result = scope["_transcribe_audio_whisper"]("test.wav", candidate_languages=["en", "ru"])
        self.assertEqual(result[0], "Actual speech")
        self.assertFalse(result[3])

    def test_language_retry_is_bounded_and_disjoint(self):
        scope = functions("segmenter.py", {"language_recovery_parts"}, {
            "np": np, "_speech_spans": lambda wav: [(0., len(wav) / 16000)],
        })
        split = scope["language_recovery_parts"]
        for seconds in (2., 3., 4., 8., 10.):
            wav = np.arange(int(seconds * 16000))
            parts = split(wav, 16000, 2.5, 1.)
            self.assertGreaterEqual(len(parts), 2)
            self.assertTrue(all(16000 <= len(p) < len(wav) for p in parts))
            np.testing.assert_array_equal(np.concatenate(parts), wav)
        self.assertEqual(split(np.zeros(24000), 16000, 2.5, 1.), [])

    def test_short_bootstrap_numbered_new_voices_and_reassignment(self):
        import threading
        scope = dict(
            np=np, time=__import__("time"), logger=logging.getLogger("test"),
            SPK_ACCEPT=.55, SPK_NEW=.4, SPK_MIN_ENROLL_SPEECH_S=2., SPK_MIN_MARGIN=.08,
            SPK_BOOTSTRAP_MIN_SPEECH_S=1.5, SPK_NEW_IMMEDIATE_PAIR_MASS=.95,
            SPK_NEW_IMMEDIATE_NEAR_RATIO=.75, SPK_ENROLL_MIN_PURITY=.6,
            SPK_NEW_MIN_PAIR_MASS=.9, SPK_NEW_MIN_NEAR_FIELD_RATIO=.5,
            SPK_NEW_CONFIRM_TURNS=2, MIN_TRIMMED_DURATION_S=.5,
            SPK_SHORT_TURN_S=2., SPK_DUPLICATE_PROTOTYPE_SIM=.95,
            SPK_BOOTSTRAP_PROTOTYPES=3, SPK_BOOTSTRAP_ACCEPT=.45, SPK_SHORT_CONTINUE_SIM=.3, SPK_SHORT_CONTINUE_MAX_SPEECH_S=3., SPK_ENROLL_MIN_SIM=.6,
            SPK_MAX_PROTOTYPES=10,
            _trim_trailing_silence=lambda wav, sr: wav,
            _trim_unknown_buffer=lambda session: None,
            _check_and_merge=lambda session: [],
            _apply_remap=lambda sid, merges: sid,
            _save_debug_audio_async=lambda *args: None,
        )
        functions("speaker_diart.py", {"identify_speaker", "_new_speaker_evidence",
                  "_speaker_best_sim", "_cosine_sim", "_add_prototype",
                  "_process_unknown_buffer", "_best_existing_match", "embed_for_match"}, scope)
        current = [np.array([1., 0., 0.])]
        scope["_embed"] = lambda *args: current[0]
        identify = scope["identify_speaker"]
        session = dict(speakers={}, next_id=1, unknown_buffer=[],
                       last_speaker=None, lock=threading.Lock())
        def turn(seq, duration, mass=.978, ratio=None):
            return identify(session, np.ones(int(duration * 16000)), 16000, seq,
                            pair_mass=mass, near_field_ratio=ratio)
        # A short, confident first turn in the supplied log now gets speaker 1.
        self.assertEqual(turn(3, 1.792)["speaker"], 1)
        self.assertEqual(turn(4, 2.5)["speaker"], 1)
        current[0] = np.array([0., 1., 0.])
        self.assertIsNone(turn(5, 1.2, ratio=1.)["speaker"])
        result = turn(6, 3., ratio=1.)
        self.assertEqual(result["speaker"], 2)
        self.assertIn((5, 2), result["reassigned"])
        current[0] = np.array([0., 0., 1.])
        self.assertEqual(turn(7, 3., ratio=1.)["speaker"], 3)
        self.assertEqual(turn(8, 3., ratio=1.)["speaker"], 3)
        # The previous false-speaker example does not qualify for immediate ID.
        current[0] = np.array([-.8, -.6, 0.])
        self.assertIsNone(turn(9, 4., mass=.929, ratio=.60)["speaker"])
        self.assertEqual(len(session["speakers"]), 3)

    def test_new_speaker_needs_independent_strong_turns(self):
        scope = functions("speaker_diart.py", {"_process_unknown_buffer", "_new_speaker_evidence"}, {
            "np": np, "logger": logging.getLogger("test"),
            "SPK_ACCEPT": .55, "SPK_MIN_ENROLL_SPEECH_S": 2.,
            "SPK_NEW_CONFIRM_TURNS": 2, "SPK_ENROLL_MIN_PURITY": .6,
            "SPK_NEW_MIN_NEAR_FIELD_RATIO": .5, "SPK_NEW_MIN_PAIR_MASS": .9,
            "_trim_unknown_buffer": lambda session: None,
            "_best_existing_match": lambda *args: (None, -1.),
            "_cosine_sim": lambda a, b: float(np.dot(a, b)),
            "_add_prototype": lambda profile, embedding, seq: profile["prototypes"].append(seq),
        })
        def entry(seq, mass=.97):
            return dict(seq=seq, embedding=np.array([1., 0.]), speech_s=3.,
                        purity=1., near_field_ratio=.7, pair_mass=mass)
        session = dict(unknown_buffer=[entry(5.001), entry(5.002)], speakers={}, next_id=2)
        process = scope["_process_unknown_buffer"]
        self.assertEqual(process(session), [])
        session["unknown_buffer"].append(entry(6, .8))
        self.assertEqual(process(session), [])
        session["unknown_buffer"].append(entry(7))
        result = process(session)
        self.assertEqual({seq for seq, _ in result}, {5.001, 5.002, 7})
        self.assertEqual([e["seq"] for e in session["unknown_buffer"]], [6])

    def test_speaker_tie_marks_uncertain_without_enrollment(self):
        import threading
        scope = dict(
            np=np, time=__import__("time"), logger=logging.getLogger("test"),
            SPK_ACCEPT=.55, SPK_NEW=.4, SPK_MIN_MARGIN=.08, SPK_MIN_ENROLL_SPEECH_S=2.,
            SPK_BOOTSTRAP_MIN_SPEECH_S=1.5, SPK_NEW_IMMEDIATE_PAIR_MASS=.95,
            SPK_NEW_IMMEDIATE_NEAR_RATIO=.75, SPK_ENROLL_MIN_PURITY=.6,
            SPK_NEW_MIN_PAIR_MASS=.9, SPK_NEW_MIN_NEAR_FIELD_RATIO=.5,
            SPK_NEW_CONFIRM_TURNS=2, MIN_TRIMMED_DURATION_S=.5,
            SPK_SHORT_TURN_S=2., SPK_DUPLICATE_PROTOTYPE_SIM=.95,
            SPK_BOOTSTRAP_PROTOTYPES=3, SPK_BOOTSTRAP_ACCEPT=.45, SPK_SHORT_CONTINUE_SIM=.3, SPK_SHORT_CONTINUE_MAX_SPEECH_S=3., SPK_ENROLL_MIN_SIM=.6,
            SPK_MAX_PROTOTYPES=10,
            _trim_trailing_silence=lambda wav, sr: wav,
            _trim_unknown_buffer=lambda session: None,
            _check_and_merge=lambda session: [],
            _apply_remap=lambda sid, merges: sid,
            _save_debug_audio_async=lambda *args: None,
        )
        functions("speaker_diart.py", {"identify_speaker", "_new_speaker_evidence",
                  "_speaker_best_sim", "_cosine_sim", "_add_prototype",
                  "_process_unknown_buffer", "_best_existing_match", "embed_for_match"}, scope)
        s = np.sqrt(1 - 0.58 ** 2)
        incoming = np.array([1., 0., 0.])
        v1 = np.array([0.58, s, 0.])
        v3 = np.array([0.58, -s, 0.])
        scope["_embed"] = lambda *args: incoming
        identify = scope["identify_speaker"]
        session = dict(
            speakers={
                1: {"prototypes": [{"seq": 1, "embedding": v1}]},
                3: {"prototypes": [{"seq": 2, "embedding": v3}]},
            },
            next_id=4, unknown_buffer=[], last_speaker=None, lock=threading.Lock(),
        )
        result = identify(session, np.ones(int(3.0 * 16000)), 16000, 51, pair_mass=.98, near_field_ratio=1.)
        self.assertEqual(result["decision"], "tie_uncertain")
        self.assertIn(result["speaker"], (1, 3))
        self.assertTrue(result["uncertain"])
        self.assertEqual(len(session["speakers"][1]["prototypes"]), 1)
        self.assertEqual(len(session["speakers"][3]["prototypes"]), 1)

if __name__ == "__main__":
    unittest.main()
