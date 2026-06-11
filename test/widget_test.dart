import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:absa_v01/main.dart';

void main() {
  testWidgets('ABSA interface validates empty review', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const AbsaApp());

    expect(find.text('Restaurant ABSA'), findsOneWidget);
    expect(find.text('Rule based'), findsOneWidget);
    expect(find.byKey(const Key('reviewInput')), findsOneWidget);
    expect(find.text('Results will appear here.'), findsOneWidget);

    await tester.tap(find.byKey(const Key('analyzeButton')));
    await tester.pump();

    expect(
      find.text('Enter a restaurant review before analyzing.'),
      findsOneWidget,
    );
  });

  testWidgets('model menu can select traditional ML', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const AbsaApp());

    await tester.tap(find.text('Rule based'));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Traditional ML').last);
    await tester.pumpAndSettle();

    expect(find.text('Traditional ML'), findsOneWidget);
  });
}
