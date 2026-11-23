# HTTP API

The API is a FastAPI service running the models on ONNX Runtime, with no PyTorch
([`backend/ridesafe/api.py`](../backend/ridesafe/api.py)). Interactive documentation, where you can try every endpoint,
is served at `/docs`: [live API docs](https://ridesafe-api.blackcliff-76dccd31.uaenorth.azurecontainerapps.io/docs)
(the live API sleeps when idle; the first request can take about 30 s).

**Contents:** [Run it](#run-it) · [Check a photo](#check-a-photo) · [Analyse a video](#analyse-a-video) ·
[Endpoints](#endpoints) · [Access and security](#access-and-security) · [Settings](#settings)

## Run it

```bash
python scripts/export_onnx.py --vehicles       # once: models/best.onnx + models/vehicles.onnx
uvicorn ridesafe.api:app --port 8000           # then open http://localhost:8000/docs
```
Or with Docker: `docker compose up --build` (see [DEPLOYMENT.md](DEPLOYMENT.md#docker)).

## Check a photo

```bash
curl -F "file=@samples/image1.jpg;type=image/jpeg" http://localhost:8000/detect
```
```json
{
  "width": 416,
  "height": 416,
  "latency_ms": 292.5,
  "with_helmet": 7,
  "without_helmet": 2,
  "detections": [
    {"cls": "With Helmet", "conf": 0.7531, "box": [115.4, 112.8, 143.0, 154.1]},
    {"cls": "With Helmet", "conf": 0.7289, "box": [198.1, 129.5, 220.1, 167.3]}
  ]
}
```
Boxes are `[x1, y1, x2, y2]` in image pixels; only detections above the tuned per-class thresholds are returned
(list shortened here). A photo is one frame: for a decision per rider, use a video.

## Analyse a video

A video is analysed in the background. Send a random `X-Owner-Token` (16-128 characters) with every request: it
makes you the owner of the upload, so only you (and the author) can see, review or delete it.

```bash
TOKEN=my-random-owner-token-1234
API=http://localhost:8000

# 1. Upload: returns 202 with the job, including its "id"
curl -H "X-Owner-Token: $TOKEN" -F "file=@traffic.mp4" $API/jobs/video

# 2. Poll until "status" is "finished" (also: "queued", "running", "failed"); "progress" counts frames
curl -H "X-Owner-Token: $TOKEN" $API/jobs/<id>

# 3. Results: every analysed frame's tracked boxes, and one decision per rider
curl -H "X-Owner-Token: $TOKEN" $API/jobs/<id>/tracks

# 4. The violations logged for this video (snapshot links, votes, plate when read)
curl -H "X-Owner-Token: $TOKEN" "$API/violations?job_id=<id>"

# 5. Review one: confirm, reject, or set back to pending
curl -X PATCH -H "X-Owner-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"review_status": "confirmed"}' $API/violations/<violation id>
```
In `/jobs/<id>/tracks`, each rider has a `label`: `violation`, `helmet`, `uncertain` (split votes, for a reviewer),
`not_riding` (a pedestrian), `cyclist` (no helmet rule) or `pending`. A rider linked to an earlier log by the
duplicate check has `duplicate_of` set. The read-only sample job (when `RIDESAFE_SAMPLE_DIR` is set) is listed for
everyone.

## Endpoints

| Method | Path | What it does | Who |
|---|---|---|---|
| GET | `/health` | liveness, and whether the models are loaded | anyone |
| GET | `/model` | classes, thresholds, input size, limits, demo mode | anyone |
| GET | `/me` | who the caller is: `author` or `visitor` | anyone |
| GET | `/metrics` | request counts and latencies, jobs, frames processed | author |
| POST | `/detect` | an image in, helmet / no-helmet boxes out (JSON) | anyone, rate-limited |
| POST | `/jobs/video` | upload a video, get an async job (202) and an owner token | anyone, rate-limited |
| GET | `/jobs` | the jobs the caller may see | anyone |
| GET | `/jobs/{id}` | status, progress, summary | whoever has the ID |
| GET | `/jobs/{id}/tracks` | per-frame tracked boxes and the final decision per rider | whoever has the ID |
| GET | `/jobs/{id}/video` | the uploaded video (HTTP range requests, so it is seekable) | whoever has the ID |
| DELETE | `/jobs/{id}` | delete a job and its results | uploader, author |
| GET | `/violations` | logged violations the caller may see (filter by job, review status) | anyone |
| GET | `/violations/{id}` | one violation | uploader, author (sample: anyone) |
| PATCH | `/violations/{id}` | set `review_status`: `pending` / `confirmed` / `rejected` | uploader, author |
| GET | `/violations/{id}/snapshot` | the violator's snapshot (JPEG, signed link) | signed link |
| GET | `/violations/{id}/plate` | the plate crop (JPEG, signed link) | author on the public site |

## Access and security

- **Author:** whoever holds `RIDESAFE_ADMIN_TOKEN`, sent as `Authorization: Bearer <key>` (the web app's
  "Author sign-in"). Sees and changes everything, sees full plates and `/metrics`, has no rate limits. The key must
  be 32+ characters, is compared in constant time, and is kept by the web app only until the tab closes.
- **Visitors:** everyone else. A browser sends one random `X-Owner-Token` with every request; a visitor sees and
  changes only the uploads made with that token, plus the read-only sample, in every mode. On the public site
  (`RIDESAFE_DEMO=true`, the default) plates are partly masked (`MP04 •• 7116`) and plate crops are not served.
- **Nothing configurable through the API:** there is no endpoint that changes the model, code or settings.
- **Audit trail:** every review change records who made it (author / uploader) and when.
- **Abuse limits:** per-visitor rate limits, a cap on waiting videos, size and length limits, uploads deleted after
  an hour. Behind a proxy the visitor is identified by the address the proxy itself adds (the last
  `X-Forwarded-For` entry), so a faked header does not get around the limits.
- **No guessable links:** other visitors' violations return 404, image links carry an HMAC signature, and job IDs
  are random 128-bit values.

Every rule has a test in [`backend/tests/test_api.py`](../backend/tests/test_api.py). The `/docs` page is public on
purpose: it describes the API and grants nothing.

## Settings

Environment variables, read by [`settings.py`](../backend/ridesafe/settings.py). All are optional; the defaults
suit a local run, and [DEPLOYMENT.md](DEPLOYMENT.md#azure-architecture) lists what the live site sets.

| Variable | Default | Meaning |
|---|---|---|
| `RIDESAFE_ADMIN_TOKEN` | none | the author's key (32+ characters); without it nobody is author |
| `RIDESAFE_DEMO` | `true` | public-site rules: plates masked and plate crops hidden for visitors |
| `RIDESAFE_SAMPLE_DIR` | none | a pre-computed, read-only sample job shown to everyone |
| `RIDESAFE_DATA_DIR` | `results/api` | uploads, job results and the SQLite log (`/data` in Docker) |
| `RIDESAFE_MODEL` / `RIDESAFE_VEHICLE_MODEL` | `models/best.onnx` / `models/vehicles.onnx` | ONNX models |
| `RIDESAFE_THRESHOLDS` / `RIDESAFE_TRACKER` | `configs/thresholds.yaml` / `configs/bytetrack.yaml` | thresholds, tracker |
| `RIDESAFE_PLATES` | `true` | read number plates (needs RapidOCR and the vehicle model) |
| `RIDESAFE_RIDING_CHECK` | `true` | only heads on a motorcycle can be violators; others are labelled `not_riding` or `cyclist` ([report](../reports/video_riding_check.md)) |
| `RIDESAFE_DEDUPE_GAP` | `60` | a violator lost for up to this many analysed frames and picked up again at the same place is linked to the first log (`duplicate_of` in the tracks), not logged twice; `0` turns it off ([report](../reports/video_duplicate_logs.md)) |
| `RIDESAFE_MAX_IMAGE_MB` / `RIDESAFE_MAX_VIDEO_MB` | `10` / `100` | upload size limits |
| `RIDESAFE_MAX_VIDEO_SECONDS` | `120` | longest video accepted |
| `RIDESAFE_MAX_QUEUED` | `10` | videos allowed to wait at once (more get 429) |
| `RIDESAFE_VID_STRIDE` | `2` | analyse every Nth video frame |
| `RIDESAFE_UPLOAD_TTL_S` | `3600` | delete uploads and their results after this many seconds (0 = keep) |
| `RIDESAFE_RATE_DETECT` / `RIDESAFE_RATE_UPLOADS` | `30` / `10` | photo checks per minute, video uploads per hour, per visitor (0 = no limit) |
| `RIDESAFE_CORS_ORIGINS` | `*` | comma-separated origins allowed to call the API |
| `RIDESAFE_CORS_ORIGIN_REGEX` | none | also allow origins matching this regex (the web app's PR previews) |
| `RIDESAFE_URL_SECRET` | random at each start | key that signs image links (set it so links survive restarts) |
| `RIDESAFE_TRUSTED_PROXY` | `false` | behind a reverse proxy: take the client address from the last `X-Forwarded-For` entry |
| `RIDESAFE_LOG_LEVEL` | `INFO` | log detail on stdout: `DEBUG`, `INFO` or `WARNING` |
