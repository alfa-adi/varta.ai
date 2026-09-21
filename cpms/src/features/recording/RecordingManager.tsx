import React from 'react';
import { RecordingLauncher } from './RecordingLauncher';
import { LiveRecordingScreen } from './LiveRecordingScreen';
import { MinimizedPill } from './MinimizedPill';
import { SessionEndReview } from './SessionEndReview';

export function RecordingManager() {
  return (
    <>
      <RecordingLauncher />
      <LiveRecordingScreen />
      <MinimizedPill />
      <SessionEndReview />
    </>
  );
}
