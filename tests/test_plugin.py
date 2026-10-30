"""离线验证模型判断、缓存隔离与安装包，不调用真实模型或 QQ。"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'astrbot_plugin_keyword_wakeup'
VERSION = 'v1.0.1'


def load_plugin(path=ROOT / 'main.py'):
    modules = {name: types.ModuleType(name) for name in
               ('astrbot', 'astrbot.api', 'astrbot.api.event', 'astrbot.api.star')}
    modules['astrbot.api'].AstrBotConfig = dict
    modules['astrbot.api'].logger = logging.getLogger('judge-test')
    def decorator(kind, **kwargs):
        def register(fn):
            fn.event_type, fn.priority = kind, kwargs.get('priority', 0)
            return fn
        return register
    modules['astrbot.api.event'].AstrMessageEvent = object
    modules['astrbot.api.event'].filter = types.SimpleNamespace(
        event_message_type=decorator, EventMessageType=types.SimpleNamespace(GROUP_MESSAGE='group'))
    class Star:
        def __init__(self, context):
            self.context = context
    modules['astrbot.api.star'].Star = Star
    modules['astrbot.api.star'].Context = object
    spec = importlib.util.spec_from_file_location('keyword_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


m = load_plugin()


class Event:
    def __init__(self, text='小助手你好', *, group='123', platform='qq1', bot='bot',
                 sender='member', private=False, wake=False, blocked=False, session=None):
        self.message_str = text
        self.group, self.platform, self.bot, self.sender = group, platform, bot, sender
        self.private = private
        self.is_at_or_wake_command = wake
        self.is_wake = True
        self.call_llm = blocked
        self.session_id = session or group
        self.unified_msg_origin = f'{platform}:GroupMessage:{self.session_id}'
        self.message_obj = types.SimpleNamespace(message=[object()])
        self.send = AsyncMock()
        self.stop_event = Mock()
    def is_private_chat(self): return self.private
    def get_group_id(self): return self.group
    def get_self_id(self): return self.bot
    def get_sender_id(self): return self.sender
    def get_sender_name(self): return '群成员'
    def get_platform_id(self): return self.platform


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = Mock()
        self.context.get_config.return_value = {'provider_settings': {'enable': True}}
        self.context.get_current_chat_provider_id = AsyncMock(return_value='current-model')
        self.context.llm_generate = AsyncMock(
            return_value=types.SimpleNamespace(completion_text='{"should_reply": true}'))
        self.config = {'group_ids': ['123', '456'], 'keywords': ['小助手', 'Bot']}
        self.plugin = m.KeywordWakeupPlugin(self.context, self.config)

    def payload(self, index=-1):
        return json.loads(self.context.llm_generate.await_args_list[index].kwargs['prompt'])

    def untouched(self, event):
        event.send.assert_not_called()
        event.stop_event.assert_not_called()
        self.assertTrue(event.is_wake)
        self.assertFalse(event.call_llm)
        self.context.conversation_manager.assert_not_called()
        self.assertEqual(event.message_obj.message.__len__(), 1)

    async def test_true_uses_default_flow_without_sending_or_replacing_message(self):
        event = Event('  你好小助手  ')
        original_chain = event.message_obj.message
        await self.plugin.on_group_message(event)
        self.assertTrue(event.is_at_or_wake_command)
        self.assertEqual(event.message_str, '  你好小助手  ')
        self.assertIs(event.message_obj.message, original_chain)
        self.untouched(event)
        self.assertEqual({call[0] for call in self.context.mock_calls},
                         {'get_config', 'get_current_chat_provider_id', 'llm_generate'})

    async def test_false_does_not_wake(self):
        self.context.llm_generate.return_value.completion_text = '{"should_reply":false}'
        event = Event()
        await self.plugin.on_group_message(event)
        self.assertFalse(event.is_at_or_wake_command)
        self.untouched(event)

    async def test_invalid_outputs_do_not_wake(self):
        for output in ('true', 'null', '[]', '{}', '{"should_reply":"true"}',
                       '{"should_reply":1}', '应该回复', 'prefix true',
                       '```json\n{"should_reply":true}\n```', None):
            with self.subTest(output=output):
                self.context.llm_generate.return_value.completion_text = output
                event = Event()
                with self.assertLogs('judge-test', level='WARNING'):
                    await self.plugin.on_group_message(event)
                self.assertFalse(event.is_at_or_wake_command)
                self.untouched(event)

    async def test_model_error_does_not_retry_or_fallback(self):
        self.plugin = m.KeywordWakeupPlugin(self.context, {**self.config, 'judge_provider_id':'small'})
        self.context.llm_generate.side_effect = RuntimeError('provider failed')
        event = Event()
        with self.assertLogs('judge-test', level='WARNING'):
            await self.plugin.on_group_message(event)
        self.assertFalse(event.is_at_or_wake_command)
        self.context.llm_generate.assert_awaited_once()
        self.context.get_current_chat_provider_id.assert_not_awaited()

    async def test_provider_lookup_failure(self):
        self.context.get_current_chat_provider_id.side_effect = RuntimeError('no model')
        event = Event()
        with self.assertLogs('judge-test', level='WARNING'):
            await self.plugin.on_group_message(event)
        self.assertFalse(event.is_at_or_wake_command)
        self.context.llm_generate.assert_not_awaited()

    async def test_timeout_cancels_judgment(self):
        cancelled = asyncio.Event()
        async def hang(**kwargs):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        self.context.llm_generate.side_effect = hang
        self.plugin.judge_timeout = 0.01
        event = Event()
        with self.assertLogs('judge-test', level='WARNING'):
            await self.plugin.on_group_message(event)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(event.is_at_or_wake_command)
        self.context.llm_generate.assert_awaited_once()

    async def test_cancellation_propagates(self):
        self.context.llm_generate.side_effect = asyncio.CancelledError
        with self.assertRaises(asyncio.CancelledError):
            await self.plugin.on_group_message(Event())

    async def test_non_candidates_never_call_model(self):
        for event in (Event(group='999'), Event(private=True), Event(sender='bot'),
                      Event('普通消息'), Event('bot'), Event(''), Event(None)):
            await self.plugin.on_group_message(event)
            self.assertFalse(event.is_at_or_wake_command)
        self.context.llm_generate.assert_not_awaited()
        self.context.get_current_chat_provider_id.assert_not_awaited()

    async def test_native_wakeup_skips_judge_and_is_cached(self):
        event = Event('原有前缀或 @ 消息', wake=True)
        await self.plugin.on_group_message(event)
        self.assertTrue(event.is_at_or_wake_command)
        self.context.llm_generate.assert_not_awaited()
        await self.plugin.on_group_message(Event())
        self.assertEqual(self.payload()['history'][0]['text'], event.message_str)

    async def test_disabled_ai_and_preexisting_veto_skip_judge(self):
        blocked = Event(blocked=True)
        await self.plugin.on_group_message(blocked)
        self.assertTrue(blocked.call_llm)
        self.context.get_config.return_value = {'provider_settings': {'enable': False}}
        await self.plugin.on_group_message(Event())
        self.context.llm_generate.assert_not_awaited()

    async def test_multiple_keywords_literal_case_and_positions(self):
        self.plugin = m.KeywordWakeupPlugin(self.context,
            {**self.config, 'keywords':['小助手', '助手', 'Bot', '.*']})
        for text in ('小助手你好', '你好小助手呀', '你好小助手', '小助手Bot助手', 'abc.*xyz'):
            before = self.context.llm_generate.await_count
            await self.plugin.on_group_message(Event(text))
            self.assertEqual(self.context.llm_generate.await_count, before + 1)
        self.assertEqual(self.payload()['matched_keywords'], ['.*'])

    async def test_explicit_model_and_custom_prompt(self):
        self.plugin = m.KeywordWakeupPlugin(self.context,
            {**self.config, 'judge_provider_id':'small', 'judge_prompt':'只在问天气时回答 {不用替换}'})
        await self.plugin.on_group_message(Event())
        kwargs = self.context.llm_generate.await_args.kwargs
        self.assertEqual(kwargs['chat_provider_id'], 'small')
        self.assertTrue(kwargs['system_prompt'].startswith('只在问天气时回答 {不用替换}'))
        self.assertIn('should_reply', kwargs['system_prompt'])
        self.assertEqual(kwargs['contexts'], [])
        self.assertIsNone(kwargs['tools'])
        self.assertEqual(set(kwargs), {'chat_provider_id','prompt','system_prompt','contexts','tools'})
        self.context.get_current_chat_provider_id.assert_not_awaited()

    async def test_current_model_uses_actual_umo(self):
        event = Event(session='member_123')
        await self.plugin.on_group_message(event)
        self.context.get_current_chat_provider_id.assert_awaited_once_with(umo=event.unified_msg_origin)
        self.assertEqual(self.context.llm_generate.await_args.kwargs['chat_provider_id'], 'current-model')

    async def test_history_limit_length_and_current_not_duplicated(self):
        self.plugin = m.KeywordWakeupPlugin(self.context, {**self.config, 'context_messages':2})
        for text in ('第一条', 'x'*3000, '最后一条'):
            await self.plugin.on_group_message(Event(text))
        await self.plugin.on_group_message(Event())
        payload = self.payload()
        self.assertEqual([item['text'] for item in payload['history']], ['x'*2000,'最后一条'])
        self.assertEqual(payload['current_message']['text'], '小助手你好')
        self.assertEqual(payload['bot_id'], 'bot')
        self.assertEqual(payload['current_message']['sender_name'], '群成员')
        self.assertIsInstance(payload['current_message']['received_at'], float)

    async def test_zero_context_retains_no_history(self):
        self.plugin = m.KeywordWakeupPlugin(self.context, {**self.config, 'context_messages':0})
        await self.plugin.on_group_message(Event('普通消息'))
        await self.plugin.on_group_message(Event())
        self.assertEqual(self.payload()['history'], [])
        self.assertEqual(self.plugin._history, {})

    async def test_group_platform_bot_isolation_and_session_sharing(self):
        for kwargs in ({'group':'456'}, {'platform':'qq2'}, {'bot':'otherbot'}):
            await self.plugin.on_group_message(Event('不应串入', **kwargs))
        await self.plugin.on_group_message(Event('同群另一个人', session='other_123'))
        await self.plugin.on_group_message(Event(session='member_123'))
        self.assertEqual([x['text'] for x in self.payload()['history']], ['同群另一个人'])

    async def test_excluded_messages_are_not_cached(self):
        for event in (Event('私聊', private=True), Event('自己', sender='bot'), Event('他群', group='999')):
            await self.plugin.on_group_message(event)
        await self.plugin.on_group_message(Event())
        self.assertEqual(self.payload()['history'], [])

    async def test_concurrent_judges_keep_their_original_snapshots(self):
        started, release = asyncio.Event(), asyncio.Event()
        captured = []
        async def judge(**kwargs):
            captured.append(json.loads(kwargs['prompt']))
            if len(captured) == 1:
                started.set()
                await release.wait()
            return types.SimpleNamespace(completion_text='{"should_reply":true}')
        self.context.llm_generate.side_effect = judge
        first = asyncio.create_task(self.plugin.on_group_message(Event('小助手第一问')))
        await started.wait()
        await self.plugin.on_group_message(Event('小助手第二问'))
        release.set()
        await first
        self.assertEqual(captured[0]['history'], [])
        self.assertEqual([x['text'] for x in captured[1]['history']], ['小助手第一问'])

    async def test_terminate_clears_history_and_prevents_pending_wakeup(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def judge(**kwargs):
            started.set()
            await release.wait()
            return types.SimpleNamespace(completion_text='{"should_reply":true}')
        self.context.llm_generate.side_effect = judge
        event = Event()
        task = asyncio.create_task(self.plugin.on_group_message(event))
        await started.wait()
        await self.plugin.terminate()
        release.set()
        await task
        self.assertFalse(event.is_at_or_wake_command)
        self.assertEqual(self.plugin._history, {})
        reloaded = m.KeywordWakeupPlugin(self.context, self.config)
        self.assertEqual(reloaded._history, {})

    async def test_disabled_and_invalid_configuration(self):
        for extra in ({'group_ids':[]}, {'keywords':[]}, {'keywords':[' ']},
                      {'context_messages':101}, {'context_messages':True},
                      {'judge_timeout':0}, {'judge_timeout':121}, {'judge_timeout':'15'},
                      {'judge_prompt':''}, {'judge_provider_id':None}, {'group_ids':[123]},
                      {'keywords':'小助手'}):
            with self.subTest(extra=extra):
                with patch.object(m.logger, 'warning'):
                    plugin = m.KeywordWakeupPlugin(self.context, {**self.config, **extra})
                event = Event()
                await plugin.on_group_message(event)
                self.assertFalse(event.is_at_or_wake_command)
        self.context.llm_generate.assert_not_awaited()

    def test_normalization_and_schema_defaults(self):
        config = {**self.config, 'group_ids':[' 123 ','123',''], 'keywords':[' 小助手 ','小助手','']}
        plugin = m.KeywordWakeupPlugin(self.context, config)
        self.assertEqual(plugin.group_ids, frozenset({'123'}))
        self.assertEqual(plugin.keywords, ('小助手',))
        self.assertIs(plugin.config, config)
        schema = json.loads((ROOT/'_conf_schema.json').read_text())
        self.assertEqual(set(schema), set(self.config) | {
            'judge_provider_id','judge_prompt','context_messages','judge_timeout'})
        self.assertEqual(schema['judge_prompt']['default'], m.DEFAULT_JUDGE_PROMPT)
        self.assertEqual(schema['judge_provider_id']['_special'], 'select_provider')
        self.assertEqual(self.plugin.on_group_message.priority, 100)

    async def test_no_continuous_wakeup(self):
        await self.plugin.on_group_message(Event())
        followup = Event('继续说')
        await self.plugin.on_group_message(followup)
        self.assertFalse(followup.is_at_or_wake_command)
        self.context.llm_generate.assert_awaited_once()

    async def test_archive_matches_source_and_loads(self):
        with ZipFile(ROOT.parent/f'{NAME}_{VERSION}.zip') as z, tempfile.TemporaryDirectory() as temp:
            names = {'main.py','_conf_schema.json','metadata.yaml','README.md','LICENSE'}
            self.assertEqual(set(z.namelist()), {f'{NAME}/{name}' for name in names})
            self.assertIsNone(z.testzip())
            for name in names:
                self.assertEqual(z.read(f'{NAME}/{name}'), (ROOT/name).read_bytes())
            self.assertIn(f'version: {VERSION}', z.read(f'{NAME}/metadata.yaml').decode())
            z.extractall(temp)
            packaged = load_plugin(Path(temp)/NAME/'main.py')
            plugin = packaged.KeywordWakeupPlugin(self.context, self.config)
            event = Event()
            await plugin.on_group_message(event)
            self.assertTrue(event.is_at_or_wake_command)


if __name__ == '__main__':
    unittest.main()
