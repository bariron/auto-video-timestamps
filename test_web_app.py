import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web_app import JobQueue, create_app, validate_submission


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.app = create_app(self.output, start_worker=False)
        self.client = self.app.test_client()
        self.queue = self.app.extensions['job_queue']

    def test_nested_disk_link_and_validation(self):
        data = validate_submission({'url': 'https://disk.yandex.ru/d/key/Lectures/one%20two.mp4'})
        self.assertEqual(data['url'], 'https://disk.yandex.ru/d/key')
        self.assertEqual(data['disk_path'], '/Lectures/one two.mp4')
        for payload in [None, [], {'url': 'https://example.org/video'}, {'url': 'file:///secret'},
                        {'url': 'https://youtu.be/abcdefghijk', 'preview': 'false'},
                        {'url': 'https://disk.yandex.ru/d/key/a.mp4', 'disk_path': '/b.mp4'}]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/api/jobs', json=payload, content_type='application/json').status_code, 400)

    def test_request_boundaries(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/', headers={'Host': 'evil.example'}).status_code, 400)
        self.assertEqual(self.client.post('/api/jobs', json={}, headers={'Origin': 'https://evil.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/jobs', data='url=x').status_code, 415)
        self.assertEqual(self.client.get('/api/jobs/unknown/log').status_code, 404)

    def test_persistent_queue_and_interrupted_jobs(self):
        response = self.client.post('/api/jobs', json={'url': 'https://youtu.be/abcdefghijk'})
        self.assertEqual(response.status_code, 202)
        job_id = response.json['id']
        self.assertEqual(JobQueue(self.output, start_worker=False).list_jobs()[0]['status'], 'queued')
        self.queue.update(job_id, status='running')
        restored = JobQueue(self.output, start_worker=False).list_jobs()[0]
        self.assertEqual(restored['status'], 'failed')
        self.assertIn('Сервер был остановлен', restored['error'])

    def test_worker_success_and_failure_preserve_transcript(self):
        for code in [0, 1]:
            with self.subTest(code=code):
                job = self.queue.submit({'url': 'https://youtu.be/abcdefghijk', 'preview': True})

                def launch(command, **kwargs):
                    directory = Path(command[command.index('--output-dir') + 1])
                    self.assertIn('300', command)
                    self.assertNotIn('--skip-chapters', command)
                    (directory / 'video_result.json').write_text('{"chunks": []}', encoding='utf-8')
                    (directory / 'video_timestamps.txt').write_text('00:00:00 Речь', encoding='utf-8')
                    if code == 0:
                        (directory / 'video_chapters.txt').write_text('00:00:00 Введение', encoding='utf-8')
                    class Process:
                        stdout = io.StringIO('Transcribing with Whisper...\nProcessing chunk 1/2\nGenerating topic chapters...\n')
                        def __enter__(self): return self
                        def __exit__(self, *args): pass
                        def wait(self): return code
                    return Process()

                with patch('web_app.subprocess.Popen', side_effect=launch):
                    self.queue.run_job(job)
                saved = next(j for j in self.queue.list_jobs() if j['id'] == job['id'])
                self.assertEqual(saved['status'], 'done' if code == 0 else 'failed')
                results = self.client.get('/api/results').json
                result = next(r for r in results if r['job_id'] == job['id'])
                detail = self.client.get('/api/results/' + result['id']).json
                self.assertIn('Речь', detail['transcript'])
                self.assertEqual('chapters' in detail, code == 0)
                self.assertIn('Processing chunk', self.client.get('/api/jobs/' + job['id'] + '/log').json['text'])

    def test_existing_artifacts_and_download_allowlist(self):
        (self.output / 'lecture_result.json').write_text('{"chunks": []}')
        (self.output / 'lecture_chapters.txt').write_text('00:00:00 <script>text</script>', encoding='utf-8')
        result = self.client.get('/api/results').json[0]
        self.assertIn('chapters', result['files'])
        self.assertIn('updated', result)
        self.assertEqual(self.client.get('/api/results/' + result['id']).json['chapters'], '00:00:00 <script>text</script>')
        response = self.client.get('/api/results/' + result['id'] + '/download/chapters')
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers['Content-Disposition'])
        response.close()
        self.assertEqual(self.client.get('/api/results/' + result['id'] + '/download/job.json').status_code, 404)
        self.assertEqual(self.client.get('/api/results/unknown').status_code, 404)


if __name__ == '__main__':
    unittest.main()
