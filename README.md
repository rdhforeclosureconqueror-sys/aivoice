<div align="center">
  <div>&nbsp;</div>
  <img src="resources/openvoicelogo.jpg" width="400"/> 

[Paper](https://arxiv.org/abs/2312.01479) |
[Website](https://research.myshell.ai/open-voice) <br> <br>
<a href="https://trendshift.io/repositories/6161" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6161" alt="myshell-ai%2FOpenVoice | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>


## aiVoice service auth contract

The Render entrypoint is `server:app` from `server.py`. The public `/speak` route accepts requests with or without `x-internal-token`; token-bearing legacy callers remain compatible. The `/tts`, `/stt`, `/whisper`, and `/internal/voice/*` routes still use local service-token auth according to `VOICE_AUTH_MODE` and configured tokens.

Public service-to-service contract for Skill World production callers:

```http
POST /speak
Content-Type: application/json
x-internal-token: <optional existing caller token>
```

### Simba Wa Ujamaa / Skill World production connection

`Simba Wa Ujamaa` (`SimbaWaUjamaa.com`) is authorized for the shared Skill World lesson voice flow. The default CORS allowlist includes both `https://simbawaujamaa.com` and `https://www.simbawaujamaa.com` for browser diagnostics or future direct browser integrations. The intended production integration remains server-to-server: the Simba/Prince backend should call `POST https://aivoice-wmrv.onrender.com/speak`, then save/cache the returned bytes and send an `audio_url` back to the browser.

Backend env for Simba/Prince:

```dotenv
SKILL_WORLD_TTS_URL=https://aivoice-wmrv.onrender.com/speak
SKILL_WORLD_TTS_TOKEN=<optional shared value for older token-bearing callers>
```

aiVoice service env relevant to this contract:

- `OPENAI_API_KEY`: exact env var name used to initialize the OpenAI client.
- `OPENAI_TTS_MODEL`: optional OpenAI speech model override; defaults to `gpt-4o-mini-tts`.
- `OPENAI_TTS_VOICE`: optional default voice override; defaults to `alloy`.
- `INTERNAL_VOICE_TOKEN`: preferred shared secret for `x-internal-token`.
- `AIVOICE_API_KEY`: legacy shared secret for `X-AIVOICE-KEY`, used only when `INTERNAL_VOICE_TOKEN` is absent.
- `VOICE_AUTH_MODE`: supports `open`, `internal`, or `strict`; public `/speak` does not require a token in any mode, while `/tts`, `/stt`, `/whisper`, and `/internal/voice/*` still enforce token rules according to this mode and configured tokens.
- `ALLOWED_ORIGINS`: optional comma-separated CORS additions. Server-to-server `/speak` calls do not require an Origin header, but browser callers must exactly match the allowlist origin.
- `OPENVOICE_UPSTREAM_URL`: reported for diagnostics, but the Render `server:app` entrypoint does not proxy `/speak` to this URL.

Exact JSON body to send to `/speak`:

```json
{
  "text": "Required non-empty lesson text to synthesize.",
  "voice": "alloy",
  "format": "mp3",
  "speed": 1.0,
  "pitch": 0.0
}
```

Request field notes:

- `text` is required and must be non-empty after trimming.
- `voice` is optional. If omitted, aiVoice uses `OPENAI_TTS_VOICE`, defaulting to `alloy`.
- `format` is optional and currently accepts `mp3` or `wav`; the default is `mp3`.
- `speed` and `pitch` are accepted for shared client compatibility. The current Render OpenAI TTS route does not pass them through to OpenAI speech synthesis.

Expected `/speak` response:

- Success: HTTP `200` with raw audio bytes, not JSON.
- `format: "mp3"`: `Content-Type: audio/mpeg`.
- `format: "wav"`: `Content-Type: audio/wav`.
- Response header includes `Cache-Control: no-store`; the caller is expected to save/cache audio if needed.
- Validation errors are JSON FastAPI errors, for example HTTP `400` when `text` is empty.
- Public `/speak` does not return auth errors for missing `x-internal-token`; protected routes can still return JSON FastAPI auth errors with safe details such as `missing_internal_token` or `invalid_internal_token`.

Auth and allowlist behavior:

- Public `/speak` accepts Simba/Prince server-to-server calls without an auth header, even when `INTERNAL_VOICE_TOKEN`, legacy `AIVOICE_API_KEY`, or a non-open `VOICE_AUTH_MODE` is configured.
- Existing `/speak` callers that already send a valid `x-internal-token` or legacy `X-AIVOICE-KEY` header continue to receive the same raw audio response.
- Protected routes (`/tts`, `/stt`, `/whisper`, and `/internal/voice/*`) still enforce configured token rules.
- Simba/Prince backend calls do not need to be added to CORS because CORS only applies to browsers. Add the backend origin to `ALLOWED_ORIGINS` only if a browser will call aiVoice directly from that origin.

Operational limits:

- The service enforces request validation and the upstream OpenAI speech API limits. No repository-local per-domain or per-site rate limiter is defined for `/speak`.
- The Render `server:app` `/speak` route has no explicit application timeout in this repository; platform and upstream provider timeouts may still apply.
- Allowed response formats are `mp3` and `wav`.
- Allowed voices are governed by the configured OpenAI TTS model. `alloy` is the default and recommended Skill World voice unless a product-specific voice is configured.

Verified curl for Simba/Prince backend no-token testing:

```bash
curl -fS \
  -X POST "${SKILL_WORLD_TTS_URL:-https://aivoice-wmrv.onrender.com/speak}" \
  -H "Content-Type: application/json" \
  --data '{"text":"Simba Wa Ujamaa voice connection test.","voice":"alloy","format":"mp3","speed":1.0,"pitch":0.0}' \
  --output simba-wa-ujamaa-test.mp3
```

For backward-compatibility testing, send the same `/speak` request with `-H "x-internal-token: ${SKILL_WORLD_TTS_TOKEN}"`; both the no-token and valid-token forms should return raw audio. `/cors-debug` may still report the configured protected-route auth header without making public `/speak` require that header.

Safe auth failure details/log reasons are `local_route_auth_failed`, `missing_internal_token`, `invalid_internal_token`, `openai_auth_failed`, and `upstream_proxy_auth_failed`.

## Introduction

### OpenVoice V1

As we detailed in our [paper](https://arxiv.org/abs/2312.01479) and [website](https://research.myshell.ai/open-voice), the advantages of OpenVoice are three-fold:

**1. Accurate Tone Color Cloning.**
OpenVoice can accurately clone the reference tone color and generate speech in multiple languages and accents.

**2. Flexible Voice Style Control.**
OpenVoice enables granular control over voice styles, such as emotion and accent, as well as other style parameters including rhythm, pauses, and intonation. 

**3. Zero-shot Cross-lingual Voice Cloning.**
Neither of the language of the generated speech nor the language of the reference speech needs to be presented in the massive-speaker multi-lingual training dataset.

### OpenVoice V2

In April 2024, we released OpenVoice V2, which includes all features in V1 and has:

**1. Better Audio Quality.**
OpenVoice V2 adopts a different training strategy that delivers better audio quality.

**2. Native Multi-lingual Support.**
English, Spanish, French, Chinese, Japanese and Korean are natively supported in OpenVoice V2.

**3. Free Commercial Use.**
Starting from April 2024, both V2 and V1 are released under MIT License. Free for commercial use.

[Video](https://github.com/myshell-ai/OpenVoice/assets/40556743/3cba936f-82bf-476c-9e52-09f0f417bb2f)

OpenVoice has been powering the instant voice cloning capability of [myshell.ai](https://app.myshell.ai/explore) since May 2023. Until Nov 2023, the voice cloning model has been used tens of millions of times by users worldwide, and witnessed the explosive user growth on the platform.

## Main Contributors

- [Zengyi Qin](https://www.qinzy.tech) at MIT
- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Ethan Sun](https://twitter.com/ethan_myshell) at MyShell

## How to Use
Please see [usage](docs/USAGE.md) for detailed instructions.

## Common Issues

Please see [QA](docs/QA.md) for common questions and answers. We will regularly update the question and answer list.

## Citation
```
@article{qin2023openvoice,
  title={OpenVoice: Versatile Instant Voice Cloning},
  author={Qin, Zengyi and Zhao, Wenliang and Yu, Xumin and Sun, Xin},
  journal={arXiv preprint arXiv:2312.01479},
  year={2023}
}
```

## License
OpenVoice V1 and V2 are MIT Licensed. Free for both commercial and research use.

## Acknowledgements
This implementation is based on several excellent projects, [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), and [VITS2](https://github.com/daniilrobnikov/vits2). Thanks for their awesome work!
