import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

void main() {
  runApp(const AbsaApp());
}

enum AbsaModel {
  ruleBased('rule_based', 'Rule based'),
  traditionalMl('traditional_ml', 'Traditional ML'),
  deepLearning('bert', 'Deep learning');

  const AbsaModel(this.id, this.label);

  final String id;
  final String label;
}

class AspectResult {
  const AspectResult({
    required this.aspect,
    required this.sentiment,
    this.confidence,
  });

  final String aspect;
  final String sentiment;
  final double? confidence;
}

class AbsaApp extends StatelessWidget {
  const AbsaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Restaurant ABSA',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF2563EB),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          filled: true,
          fillColor: Color(0xFFF8FAFC),
        ),
      ),
      home: const AbsaHomePage(),
    );
  }
}

class AbsaHomePage extends StatefulWidget {
  const AbsaHomePage({super.key});

  @override
  State<AbsaHomePage> createState() => _AbsaHomePageState();
}

class _AbsaHomePageState extends State<AbsaHomePage> {
  final TextEditingController _reviewController = TextEditingController();
  final AbsaApiClient _apiClient = const AbsaApiClient();
  AbsaModel _selectedModel = AbsaModel.ruleBased;
  List<AspectResult> _results = const [];
  String? _message;
  bool _isLoading = false;

  @override
  void dispose() {
    _reviewController.dispose();
    super.dispose();
  }

  Future<void> _analyzeReview() async {
    final review = _reviewController.text.trim();
    if (review.isEmpty) {
      setState(() {
        _message = 'Enter a restaurant review before analyzing.';
        _results = const [];
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _message = null;
      _results = const [];
    });

    try {
      final results = await _apiClient.analyze(
        review: review,
        method: _selectedModel.id,
      );
      if (!mounted) return;
      setState(() {
        _results = results;
        _message = null;
      });
    } on Exception catch (error) {
      if (!mounted) return;
      setState(() {
        _message =
            'Backend unavailable or returned an error. Start the API with: python3 backends/api/app.py\n\n$error';
        _results = const [];
      });
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF1F5F9),
      appBar: AppBar(
        titleSpacing: 12,
        title: _ModelMenuButton(
          selectedModel: _selectedModel,
          onSelected: (model) {
            setState(() {
              _selectedModel = model;
              _results = const [];
              _message = null;
            });
          },
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: Center(
              child: Text(
                'Restaurant ABSA',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700),
              ),
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: LayoutBuilder(
            builder: (context, constraints) {
              final compact = constraints.maxWidth < 720;
              final content = compact
                  ? Column(
                      children: [
                        Expanded(
                          child: _InputPanel(
                            controller: _reviewController,
                            selectedModel: _selectedModel,
                            isLoading: _isLoading,
                            onAnalyze: _analyzeReview,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Expanded(
                          child: _OutputPanel(
                            selectedModel: _selectedModel,
                            message: _message,
                            isLoading: _isLoading,
                            results: _results,
                          ),
                        ),
                      ],
                    )
                  : Row(
                      children: [
                        Expanded(
                          child: _InputPanel(
                            controller: _reviewController,
                            selectedModel: _selectedModel,
                            isLoading: _isLoading,
                            onAnalyze: _analyzeReview,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _OutputPanel(
                            selectedModel: _selectedModel,
                            message: _message,
                            isLoading: _isLoading,
                            results: _results,
                          ),
                        ),
                      ],
                    );

              return content;
            },
          ),
        ),
      ),
    );
  }
}

class _ModelMenuButton extends StatelessWidget {
  const _ModelMenuButton({
    required this.selectedModel,
    required this.onSelected,
  });

  final AbsaModel selectedModel;
  final ValueChanged<AbsaModel> onSelected;

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<AbsaModel>(
      tooltip: 'Choose model',
      initialValue: selectedModel,
      onSelected: onSelected,
      itemBuilder: (context) {
        return [
          for (final model in AbsaModel.values)
            PopupMenuItem(
              value: model,
              child: Row(
                children: [
                  Icon(
                    model == selectedModel
                        ? Icons.radio_button_checked
                        : Icons.radio_button_unchecked,
                    size: 18,
                  ),
                  const SizedBox(width: 10),
                  Text(model.label),
                ],
              ),
            ),
        ];
      },
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: const Color(0xFFEFF6FF),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: const Color(0xFFBFDBFE)),
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.psychology_alt_outlined, size: 20),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  selectedModel.label,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontWeight: FontWeight.w700),
                ),
              ),
              const SizedBox(width: 4),
              const Icon(Icons.keyboard_arrow_down, size: 20),
            ],
          ),
        ),
      ),
    );
  }
}

class _InputPanel extends StatelessWidget {
  const _InputPanel({
    required this.controller,
    required this.selectedModel,
    required this.isLoading,
    required this.onAnalyze,
  });

  final TextEditingController controller;
  final AbsaModel selectedModel;
  final bool isLoading;
  final VoidCallback onAnalyze;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(Icons.rate_review_outlined),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Input review',
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Expanded(
            child: TextField(
              key: const Key('reviewInput'),
              controller: controller,
              expands: true,
              maxLines: null,
              minLines: null,
              textAlignVertical: TextAlignVertical.top,
              decoration: const InputDecoration(
                hintText:
                    'Example: The food was delicious, but the service was slow.',
              ),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            key: const Key('analyzeButton'),
            onPressed: isLoading ? null : onAnalyze,
            icon: isLoading
                ? const SizedBox.square(
                    dimension: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.analytics_outlined),
            label: Text(
              isLoading
                  ? 'Analyzing...'
                  : 'Analyze with ${selectedModel.label}',
            ),
          ),
        ],
      ),
    );
  }
}

class _OutputPanel extends StatelessWidget {
  const _OutputPanel({
    required this.selectedModel,
    required this.message,
    required this.isLoading,
    required this.results,
  });

  final AbsaModel selectedModel;
  final String? message;
  final bool isLoading;
  final List<AspectResult> results;

  @override
  Widget build(BuildContext context) {
    return _Panel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(Icons.summarize_outlined),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Output',
                  style: Theme.of(
                    context,
                  ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800),
                ),
              ),
              Text(
                selectedModel.id,
                style: Theme.of(context).textTheme.labelLarge?.copyWith(
                  color: const Color(0xFF475569),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Expanded(
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: const Color(0xFFF8FAFC),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: const Color(0xFFE2E8F0)),
              ),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: _buildOutputContent(context),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildOutputContent(BuildContext context) {
    if (isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (message != null) {
      return Center(
        child: Text(
          message!,
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodyLarge?.copyWith(
            color: const Color(0xFFB91C1C),
            fontWeight: FontWeight.w700,
          ),
        ),
      );
    }

    if (results.isEmpty) {
      return Center(
        child: Text(
          'Results will appear here.',
          style: Theme.of(
            context,
          ).textTheme.bodyLarge?.copyWith(color: const Color(0xFF64748B)),
        ),
      );
    }

    return ListView.separated(
      itemCount: results.length,
      separatorBuilder: (context, index) => const SizedBox(height: 10),
      itemBuilder: (context, index) {
        final result = results[index];
        return _ResultRow(result: result);
      },
    );
  }
}

class AbsaApiClient {
  const AbsaApiClient({
    this.baseUrl = const String.fromEnvironment(
      'ABSA_API_BASE',
      defaultValue: 'http://127.0.0.1:8000',
    ),
  });

  final String baseUrl;

  Future<List<AspectResult>> analyze({
    required String review,
    required String method,
  }) async {
    final client = HttpClient();
    try {
      final uri = Uri.parse('$baseUrl/analyze');
      final request = await client
          .postUrl(uri)
          .timeout(const Duration(seconds: 8));
      final requestBody = utf8.encode(
        jsonEncode({'review': review, 'method': method}),
      );
      request.headers.contentType = ContentType.json;
      request.contentLength = requestBody.length;
      request.add(requestBody);

      final response = await request.close().timeout(
        const Duration(seconds: 20),
      );
      final responseBody = await response.transform(utf8.decoder).join();
      final decoded = jsonDecode(responseBody) as Map<String, dynamic>;

      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(decoded['error'] ?? 'HTTP ${response.statusCode}');
      }

      final rawResults = decoded['results'] as List<dynamic>;
      return [
        for (final item in rawResults.cast<Map<String, dynamic>>())
          AspectResult(
            aspect: item['aspect'] as String,
            sentiment: item['sentiment'] as String,
            confidence: (item['confidence'] as num?)?.toDouble(),
          ),
      ];
    } on TimeoutException {
      throw Exception('Request timed out while connecting to $baseUrl.');
    } on SocketException {
      throw Exception('Cannot connect to $baseUrl.');
    } finally {
      client.close(force: true);
    }
  }
}

class _ResultRow extends StatelessWidget {
  const _ResultRow({required this.result});

  final AspectResult result;

  @override
  Widget build(BuildContext context) {
    final color = switch (result.sentiment) {
      'Positive' => const Color(0xFF15803D),
      'Negative' => const Color(0xFFB91C1C),
      _ => const Color(0xFF64748B),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 11),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              result.aspect,
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
          ),
          const SizedBox(width: 12),
          DecoratedBox(
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(999),
              border: Border.all(color: color.withValues(alpha: 0.28)),
            ),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              child: Text(
                result.sentiment,
                style: TextStyle(color: color, fontWeight: FontWeight.w800),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Panel extends StatelessWidget {
  const _Panel({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFE2E8F0)),
        boxShadow: const [
          BoxShadow(
            color: Color(0x140F172A),
            blurRadius: 16,
            offset: Offset(0, 6),
          ),
        ],
      ),
      child: child,
    );
  }
}
