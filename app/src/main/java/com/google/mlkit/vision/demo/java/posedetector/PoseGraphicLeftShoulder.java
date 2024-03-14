/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.posedetector;

import static java.lang.Math.max;
import static java.lang.Math.min;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.GraphicOverlay.Graphic;

import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;
import java.util.List;
import java.util.Locale;
import android.graphics.RectF;
import android.os.Build;
import androidx.annotation.RequiresApi;



/** Draw the detected pose in preview. */
public class PoseGraphicLeftShoulder extends Graphic {

    private static final float DOT_RADIUS = 8.0f;
    private static final float IN_FRAME_LIKELIHOOD_TEXT_SIZE = 30.0f;
    private static final float STROKE_WIDTH = 10.0f;
    private static final float POSE_CLASSIFICATION_TEXT_SIZE = 60.0f;

    private final Pose pose;
    private final boolean showInFrameLikelihood;
    private final boolean visualizeZ;
    private final boolean rescaleZForVisualization;
    private float zMin = Float.MAX_VALUE;
    private float zMax = Float.MIN_VALUE;

    private final List<String> poseClassification;
    private final Paint classificationTextPaint;
    private final Paint leftPaint;
//    private final Paint rightPaint;
    private final Paint whitePaint;

    PoseGraphicLeftShoulder(
            GraphicOverlay overlay,
            Pose pose,
            boolean showInFrameLikelihood,
            boolean visualizeZ,
            boolean rescaleZForVisualization,
            List<String> poseClassification) {
        super(overlay);
        this.pose = pose;
        this.showInFrameLikelihood = showInFrameLikelihood;
        this.visualizeZ = visualizeZ;
        this.rescaleZForVisualization = rescaleZForVisualization;

        this.poseClassification = poseClassification;
        classificationTextPaint = new Paint();
        classificationTextPaint.setColor(Color.WHITE);
        classificationTextPaint.setTextSize(POSE_CLASSIFICATION_TEXT_SIZE);
        classificationTextPaint.setShadowLayer(5.0f, 0f, 0f, Color.BLACK);

        whitePaint = new Paint();
        whitePaint.setStrokeWidth(STROKE_WIDTH);
        whitePaint.setColor(Color.WHITE);
        whitePaint.setTextSize(IN_FRAME_LIKELIHOOD_TEXT_SIZE);
        leftPaint = new Paint();
        leftPaint.setStrokeWidth(STROKE_WIDTH);
        leftPaint.setColor(Color.GREEN);
        }



    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    public void draw(Canvas canvas) {
        List<PoseLandmark> landmarks = pose.getAllPoseLandmarks();
        if (landmarks.isEmpty()) {
            return;
        }

        // Draw pose classification text.
        float classificationX = POSE_CLASSIFICATION_TEXT_SIZE * 0.5f;
        for (int i = 0; i < poseClassification.size(); i++) {
            float classificationY =
                    (canvas.getHeight()
                            - POSE_CLASSIFICATION_TEXT_SIZE * 1.5f * (poseClassification.size() - i));
            canvas.drawText(
                    poseClassification.get(i), classificationX, classificationY, classificationTextPaint);
        }

        // Draw all the points
        for (PoseLandmark landmark : landmarks) {
            if(landmark.getLandmarkType() == PoseLandmark.LEFT_SHOULDER ||
                    landmark.getLandmarkType() == PoseLandmark.LEFT_ELBOW ||
                    landmark.getLandmarkType() == PoseLandmark.LEFT_HIP ) {

                drawPoint(canvas, landmark, whitePaint);
                if (visualizeZ && rescaleZForVisualization) {
                    zMin = min(zMin, landmark.getPosition3D().getZ());
                    zMax = max(zMax, landmark.getPosition3D().getZ());
                }
            }
        }


        PoseLandmark leftShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER);
        PoseLandmark leftElbow = pose.getPoseLandmark(PoseLandmark.LEFT_ELBOW);
        PoseLandmark leftHip = pose.getPoseLandmark(PoseLandmark.LEFT_HIP);

        if (leftShoulder == null || leftElbow == null || leftHip == null)
            return;

        // Left body
        drawLine(canvas, leftShoulder, leftElbow, leftPaint);
        drawLine(canvas, leftShoulder, leftHip, leftPaint);

        double angle = calculateAngle(leftElbow, leftShoulder, leftHip);
        drawArc(canvas, leftElbow, leftShoulder, angle);

        // Draw angle for leftShoulder
        canvas.drawText(
                        String.format(Locale.US, "%.0f", angle),
                        translateX(leftShoulder.getPosition().x),
                        translateY(leftShoulder.getPosition().y),
                        whitePaint);

        // Draw inFrameLikelihood for all points
        if (showInFrameLikelihood) {
            for (PoseLandmark landmark : landmarks) {
                canvas.drawText(
                        String.format(Locale.US, "%.2f", landmark.getInFrameLikelihood()),
                        translateX(landmark.getPosition().x),
                        translateY(landmark.getPosition().y),
                        whitePaint);
            }
        }
    }

    void drawPoint(Canvas canvas, PoseLandmark landmark, Paint paint) {
        PointF3D point = landmark.getPosition3D();
        updatePaintColorByZValue(
                paint, canvas, visualizeZ, rescaleZForVisualization, point.getZ(), zMin, zMax);
        canvas.drawCircle(translateX(point.getX()), translateY(point.getY()), DOT_RADIUS, paint);
    }

    double calculateAngle(PoseLandmark pointA, PoseLandmark pointB, PoseLandmark pointC) {
        double vector1X = pointA.getPosition().x - pointB.getPosition().x;
        double vector1Y = pointA.getPosition().y - pointB.getPosition().y;
        double vector2X = pointC.getPosition().x - pointB.getPosition().x;
        double vector2Y = pointC.getPosition().y - pointB.getPosition().y;

        double dotProduct = vector1X * vector2X + vector1Y * vector2Y;
        double magnitude1 = Math.sqrt(vector1X * vector1X + vector1Y * vector1Y);
        double magnitude2 = Math.sqrt(vector2X * vector2X + vector2Y * vector2Y);

        double cosine = dotProduct / (magnitude1 * magnitude2);
        double angle = Math.acos(cosine);

        // Convert angle to degrees
        angle = Math.toDegrees(angle);

        return angle;
    }

    void drawArc(Canvas canvas, PoseLandmark pointA, PoseLandmark pointB, double angle){

        // Calculate the radius (distance from midpoint to any of the end points)
//        double radius = Math.sqrt(Math.pow(pointB.getPosition().x - pointA.getPosition().x, 2) + Math.pow(pointB.getPosition().y - pointA.getPosition().y, 2));

        PointF3D pa = pointA.getPosition3D();
        PointF3D pb = pointB.getPosition3D();

        float xa = translateX(pa.getX());
        float ya = translateY(pa.getY());

        float xb = translateX(pb.getX());
        float yb = translateY(pb.getY());

        float radius = (float)Math.sqrt(Math.pow(xa- xb, 2) + Math.pow(ya - yb, 2))/2;

        // Calculate the start and sweep angles
        double startAngle = Math.toDegrees(Math.atan2(ya - yb, xa - xb));

        Paint p = new Paint();
        p.setColor(Color.BLACK);
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(5f);

//        // Draw the arc
        RectF arcBounds = new RectF((xb - radius), (yb - radius), (xb + radius), (yb + radius));
        canvas.drawArc(arcBounds, (float) startAngle, (float) (angle), false, p);
    }


        void drawLine(Canvas canvas, PoseLandmark startLandmark, PoseLandmark endLandmark, Paint paint) {
        PointF3D start = startLandmark.getPosition3D();
        PointF3D end = endLandmark.getPosition3D();

        // Gets average z for the current body line
        float avgZInImagePixel = (start.getZ() + end.getZ()) / 2;
        updatePaintColorByZValue(
                paint, canvas, visualizeZ, rescaleZForVisualization, avgZInImagePixel, zMin, zMax);

        canvas.drawLine(
                translateX(start.getX()),
                translateY(start.getY()),
                translateX(end.getX()),
                translateY(end.getY()),
                paint);
    }
}
