import json
import unittest

from generate_chapters import build_detail_batches, generate_detailed, validate_detail_output, generation_signature


class DetailedChapterTests(unittest.TestCase):
    def test_full_long_transcript_is_partitioned_without_loss(self):
        segments = [{'start': i * 15, 'text': f'Unique sentence {i}. ' * 30} for i in range(640)]
        batches = build_detail_batches(segments, max_chars=4000)
        flattened = [s for batch in batches for s in batch]
        self.assertEqual([s['text'] for s in flattened], [s['text'] for s in segments])
        self.assertEqual([s['segment_id'] for s in flattened], list(range(640)))
        self.assertEqual(flattened[-1]['start'], 9585)
        self.assertTrue(all(sum(len(s['text']) + 40 for s in batch) <= 4000 for batch in batches))

    def test_chapters_cover_all_batches_without_global_ten_chapter_limit(self):
        batches = build_detail_batches([{'start': i * 60, 'text': f'Topic {i}'} for i in range(160)])
        seen = []

        def generate(prompt):
            batch = batches[len(seen)]
            seen.append(prompt)
            return json.dumps([{'segment_id': s['segment_id'], 'title': 'Specific mathematical proof technique'} for s in (batch[0], batch[-1])])

        chapters = generate_detailed(batches, generate)
        self.assertEqual(len(seen), len(batches))
        self.assertGreater(len(chapters), 10)
        self.assertEqual(chapters[-1]['start'], '02:39:00')
        self.assertEqual(chapters[-1]['source_text'], 'Topic 159')

    def test_rejects_fabricated_ids_generic_titles_and_missing_tail(self):
        batch = [{'segment_id': i, 'start': i * 60, 'text': 'Some actual speech'} for i in range(8)]
        for items in [
            [{'segment_id': 99, 'title': 'An invented topic name'}],
            [{'segment_id': 0, 'title': 'Введение'}],
            [{'segment_id': 0, 'title': 'Инструменты и материалы'}],
            [{'segment_id': 0, 'title': 'A concrete first topic'}, {'segment_id': 1, 'title': 'A concrete second topic'}],
            [{'segment_id': 0, 'title': 'A concrete first topic'}, {'segment_id': 0, 'title': 'A duplicated first topic'}],
        ]:
            with self.subTest(items=items), self.assertRaises(ValueError):
                validate_detail_output(json.dumps(items), batch, first_batch=True)

    def test_retry_invalid_model_response_and_fail_instead_of_partial_success(self):
        batch = [{'segment_id': 0, 'start': 0, 'text': 'Source speech'}]
        replies = iter(['not JSON', '[{"segment_id": 0, "title": "A precise topic title"}]'])
        trace = []
        self.assertEqual(len(generate_detailed([batch], lambda _: next(replies), trace)), 1)
        self.assertEqual(len(trace), 2)
        with self.assertRaisesRegex(ValueError, 'batch 1'):
            generate_detailed([batch], lambda _: '[]')

    def test_reuses_only_valid_checkpoint_batches(self):
        batch = [{'segment_id': 0, 'start': 0, 'text': 'Source speech'}]
        trace = [{'batch': 1, 'output': '[{"segment_id": 0, "title": "A precise topic title"}]'}]
        def unexpected_call(_):
            self.fail('A verified cached batch should not reload inference')
        self.assertEqual(len(generate_detailed([batch], unexpected_call, trace)), 1)
        first = generation_signature([{'text': 'original'}], 'model', [batch])
        self.assertNotEqual(first, generation_signature([{'text': 'changed'}], 'model', [batch]))
        self.assertNotEqual(first, generation_signature([{'text': 'original'}], 'different-model', [batch]))


if __name__ == '__main__':
    unittest.main()
