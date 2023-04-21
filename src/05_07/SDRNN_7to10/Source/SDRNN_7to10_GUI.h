/*
  ==============================================================================

   This file is part of the JUCE tutorials.
   Copyright (c) 2020 - Raw Material Software Limited

   The code included in this file is provided under the terms of the ISC license
   http://www.isc.org/downloads/software-support-policy/isc-license. Permission
   To use, copy, modify, and/or distribute this software for any purpose with or
   without fee is hereby granted provided that the above copyright notice and
   this permission notice appear in all copies.

   THE SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, AND ALL WARRANTIES,
   WHETHER EXPRESSED OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR
   PURPOSE, ARE DISCLAIMED.

  ==============================================================================
*/

/*******************************************************************************
 The block below describes the properties of this PIP. A PIP is a short snippet
 of code that can be read by the Projucer and used to generate a JUCE project.

 BEGIN_JUCE_PIP_METADATA

 name:             SDRNN_7to10
 version:          1.0.0
 vendor:           JUCE
 website:          http://juce.com
 description:      Segment Display Recognition Neural Network (7 to 10 model).

 dependencies:     juce_core, juce_data_structures, juce_events, juce_graphics,
                   juce_gui_basics
 exporters:        xcode_mac, vs2019, linux_make

 type:             Component
 mainClass:        MainContentComponent

 useLocalCopy:     1

 END_JUCE_PIP_METADATA

*******************************************************************************/


#pragma once
#include "MLP.h"
#include <string>
#include <iterator>

#define WIN_W 500
#define WIN_H 430

#define O1X  20
#define O1Y  20
#define VSW 72
#define VSL 100
#define HSW 50
#define HSL 100
#define V1X 0+O1X
#define V2X V1X+VSW+VSL
#define H1Y 0+O1Y
#define H2Y H1Y+HSW+VSL-15
#define H3Y H2Y+HSW+VSL-15
#define SLaX V1X+VSW 
#define SLaY H1Y
#define SLbX V2X
#define SLbY H1Y+HSW
#define SLcX V2X
#define SLcY H2Y+HSW
#define SLdX V1X+VSW 
#define SLdY H3Y
#define SLeX V1X
#define SLeY H2Y+HSW
#define SLfX V1X
#define SLfY H1Y+HSW
#define SLgX V1X+VSW 
#define SLgY H2Y
#define SEGW 25
#define SEGL 100
#define aX SLaX
#define aY SLaY+((HSW*5)/12)
#define bX SLbX
#define bY SLbY
#define cX SLcX
#define cY SLcY
#define dX SLdX
#define dY SLdY+((HSW*5)/12)
#define eX SLeX+((VSW*5)/8)
#define eY SLeY
#define fX SLfX+((VSW*5)/8)
#define fY SLfY
#define gX SLgX
#define gY SLgY+((HSW*5)/12)

#define OFFSET 230

//==============================================================================
class MainContentComponent : public juce::Component,
                             public juce::Slider::Listener,
                             public juce::Button::Listener,
                             public juce::TextEditor::Listener{
public:
    //==============================================================================
    MainContentComponent(){
        srand(time(NULL));
        rand();
        sdrnn = new MultiLayerPerceptron({7,7,10});
        LookAndFeel *l = &getLookAndFeel();
        l->setColour(juce::Slider::thumbColourId,juce::Colour(110,110,110));
        l->setColour(juce::Slider::textBoxOutlineColourId,Colour(240,240,240));
        l->setColour(juce::Slider::textBoxTextColourId,juce::Colours::black);
        l->setColour(juce::Slider::backgroundColourId,juce::Colour(OFFSET,OFFSET,OFFSET));
        l->setColour(juce::Slider::trackColourId, juce::Colour(OFFSET, OFFSET, OFFSET));
        l->setColour(juce::TextButton::buttonColourId, juce::Colour(OFFSET, OFFSET, OFFSET));
        l->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
        l->setColour(juce::TextButton::textColourOnId, juce::Colours::black);
        l->setColour(juce::Label::textColourId, juce::Colours::black);
        l->setColour(juce::Label::backgroundColourId, juce::Colour(240,240,240));
        l->setColour(juce::TextEditor::ColourIds::textColourId, juce::Colours::black);
        l->setColour(juce::TextEditor::ColourIds::backgroundColourId, juce::Colours::white);

        addAndMakeVisible(lbl_epochs_txt);
        addAndMakeVisible(entry_epochs);
        lbl_epochs_txt.setText("Epochs to Train:", no);
        entry_epochs.setText("10");
        
        addAndMakeVisible(btn_train);
        btn_train.setButtonText("Train some more");
        btn_train.addListener(this);

        addAndMakeVisible(lbl_err_txt);
        addAndMakeVisible(lbl_err);
        lbl_err_txt.setText("Training Error:",no);
        lbl_err.setText("---",no);
        lbl_err.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(lbl_tepochs_txt);
        addAndMakeVisible(lbl_tepochs);
        lbl_tepochs_txt.setText("Epochs so far:",no);
        lbl_tepochs.setText("0",no);
        lbl_tepochs.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(lbl_out0);
        addAndMakeVisible(lbl_out0_txt);
        lbl_out0_txt.setText("Output 0:", no);
        lbl_out0.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out1);
        addAndMakeVisible(lbl_out1_txt);
        lbl_out1_txt.setText("Output 1:", no);
        lbl_out1.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out2);
        addAndMakeVisible(lbl_out2_txt);
        lbl_out2_txt.setText("Output 2:", no);
        lbl_out2.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out3);
        addAndMakeVisible(lbl_out3_txt);
        lbl_out3_txt.setText("Output 3:", no);
        lbl_out3.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out4);
        addAndMakeVisible(lbl_out4_txt);
        lbl_out4_txt.setText("Output 4:", no);
        lbl_out4.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out5);
        addAndMakeVisible(lbl_out5_txt);
        lbl_out5_txt.setText("Output 5:", no);
        lbl_out5.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out6);
        addAndMakeVisible(lbl_out6_txt);
        lbl_out6_txt.setText("Output 6:", no);
        lbl_out6.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out7);
        addAndMakeVisible(lbl_out7_txt);
        lbl_out7_txt.setText("Output 7:", no);
        lbl_out7.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out8);
        addAndMakeVisible(lbl_out8_txt);
        lbl_out8_txt.setText("Output 8:", no);
        lbl_out8.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(lbl_out9);
        addAndMakeVisible(lbl_out9_txt);
        lbl_out9_txt.setText("Output 9:", no);
        lbl_out9.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(lbl_int);
        addAndMakeVisible(lbl_int_txt);
        lbl_int_txt.setText("Number Output:", no);
        lbl_int.setFont(60);
        lbl_int.setJustificationType(juce::Justification::centred);

        addAndMakeVisible(btn_reset);
        btn_reset.setButtonText("Reset");
        btn_reset.addListener(this);

        addAndMakeVisible (slider_a);
        slider_a.setRange (0.0, 1.0);
        slider_a.addListener (this);
        slider_a.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_a.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_b);
        slider_b.setRange(0.0,1.0);
        slider_b.addListener(this);
        slider_b.setSliderStyle(juce::Slider::LinearVertical);
        slider_b.setTextBoxStyle(juce::Slider::TextBoxRight,true,60,20);
        slider_b.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_c);
        slider_c.setRange(0.0,1.0);
        slider_c.addListener(this);
        slider_c.setSliderStyle(juce::Slider::LinearVertical);
        slider_c.setTextBoxStyle(juce::Slider::TextBoxRight,true,60,20);
        slider_c.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_d);
        slider_d.setRange(0.0,1.0);
        slider_d.addListener(this);
        slider_d.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_d.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_e);
        slider_e.setRange(0.0,1.0);
        slider_e.addListener(this);
        slider_e.setSliderStyle(juce::Slider::LinearVertical);
        slider_e.setTextBoxStyle(juce::Slider::TextBoxLeft,true,60,20);
        slider_e.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_f);
        slider_f.setRange(0.0,1.0);
        slider_f.addListener(this);
        slider_f.setSliderStyle(juce::Slider::LinearVertical);
        slider_f.setTextBoxStyle(juce::Slider::TextBoxLeft,true,60,20);
        slider_f.setNumDecimalPlacesToDisplay(2);

        addAndMakeVisible(slider_g);
        slider_g.setRange(0.0,1.0);
        slider_g.addListener(this);
        slider_g.setTextBoxStyle(juce::Slider::TextBoxAbove,true,60,20);
        slider_g.setNumDecimalPlacesToDisplay(2);

        setSize(WIN_W, WIN_H);
        run_ann();
    }


#define DX   90
#define DY   30
#define dy   20
#define LEN  90
#define HEI  25 
#define X0  290
#define X1 (X0+DX)
#define Y0   30
#define Y1 (Y0+DY)
#define Y2 (Y1+DY)
#define Y3 (Y2+DY)
#define Y4 (Y3+DY)
#define Y5 (Y4+dy)
#define Y6 (Y5+dy)
#define Y7 (Y6+dy)
#define Y8 (Y7+dy)
#define Y9 (Y8+dy)
#define Y10 (Y9+dy)
#define Y11 (Y10+dy)
#define Y12 (Y11+dy)
#define Y13 (Y12+dy)
#define Y14 (Y13+DY)
#define YN (Y14-10)
#define YR (Y14+DY)

    void resized() override{
        slider_a.setBounds(SLaX,SLaY,HSL,HSW);
        slider_b.setBounds(SLbX,SLbY,VSW,VSL);
        slider_c.setBounds(SLcX,SLcY,VSW,VSL);
        slider_d.setBounds(SLdX,SLdY,HSL,HSW);
        slider_e.setBounds(SLeX,SLeY,VSW,VSL);
        slider_f.setBounds(SLfX,SLfY,VSW,VSL);
        slider_g.setBounds(SLgX,SLgY,HSL,HSW);

        lbl_epochs_txt.setBounds  (X0, Y0, LEN, HEI);
        entry_epochs.setBounds    (X1, Y0, LEN, HEI);
        btn_train.setBounds       (X1, Y1, LEN, HEI);
        lbl_err_txt.setBounds     (X0, Y2, LEN, HEI);
        lbl_err.setBounds         (X1, Y2, LEN, HEI);
        lbl_tepochs_txt.setBounds (X0, Y3, LEN, HEI);
        lbl_tepochs.setBounds     (X1, Y3, LEN, HEI);
        lbl_out0_txt.setBounds    (X0, Y4, LEN, HEI);
        lbl_out0.setBounds        (X1, Y4, LEN, HEI);
        lbl_out1_txt.setBounds    (X0, Y5, LEN, HEI);
        lbl_out1.setBounds        (X1, Y5, LEN, HEI);
        lbl_out2_txt.setBounds    (X0, Y6, LEN, HEI);
        lbl_out2.setBounds        (X1, Y6, LEN, HEI);
        lbl_out3_txt.setBounds    (X0, Y7, LEN, HEI);
        lbl_out3.setBounds        (X1, Y7, LEN, HEI);
        lbl_out4_txt.setBounds    (X0, Y8, LEN, HEI);
        lbl_out4.setBounds        (X1, Y8, LEN, HEI);
        lbl_out5_txt.setBounds    (X0, Y9, LEN, HEI);
        lbl_out5.setBounds        (X1, Y9, LEN, HEI);
        lbl_out6_txt.setBounds    (X0, Y10, LEN, HEI);
        lbl_out6.setBounds        (X1, Y10, LEN, HEI);
        lbl_out7_txt.setBounds    (X0, Y11, LEN, HEI);
        lbl_out7.setBounds        (X1, Y11, LEN, HEI);
        lbl_out8_txt.setBounds    (X0, Y12, LEN, HEI);
        lbl_out8.setBounds        (X1, Y12, LEN, HEI);
        lbl_out9_txt.setBounds    (X0, Y13, LEN, HEI);
        lbl_out9.setBounds        (X1, Y13, LEN, HEI);
        lbl_int_txt.setBounds     (X0, Y14, LEN, HEI);
        lbl_int.setBounds         (X1, YN, LEN, 50);
        btn_reset.setBounds       (X0, YR, LEN, HEI);
       
        setSize(WIN_W,WIN_H);
    }

    void sliderValueChanged (juce::Slider* slider) override {
        Colour c(    (int)(slider->getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider->getValue()* OFFSET),
            OFFSET - (int)(slider->getValue()* OFFSET)    );

        slider->setColour(juce::Slider::backgroundColourId,c);
        slider->setColour(juce::Slider::trackColourId,c);
        run_ann();
        repaint();
    }

    void run_ann(){
        std::vector <double> x;
        x.push_back(slider_a.getValue());
        x.push_back(slider_b.getValue());
        x.push_back(slider_c.getValue());
        x.push_back(slider_d.getValue());
        x.push_back(slider_e.getValue());
        x.push_back(slider_f.getValue());
        x.push_back(slider_g.getValue());
        std::vector <double> out = sdrnn->run(x);
        lbl_out0.setText(std::to_string(out[0]),no);
        lbl_out1.setText(std::to_string(out[1]),no);
        lbl_out2.setText(std::to_string(out[2]),no);
        lbl_out3.setText(std::to_string(out[3]),no);
        lbl_out4.setText(std::to_string(out[4]),no);
        lbl_out5.setText(std::to_string(out[5]),no);
        lbl_out6.setText(std::to_string(out[6]),no);
        lbl_out7.setText(std::to_string(out[7]),no);
        lbl_out8.setText(std::to_string(out[8]),no);
        lbl_out9.setText(std::to_string(out[9]),no);
        lbl_int.setText(std::to_string(distance(out.begin(),
                                  max_element(out.begin(),out.end()))),no);
    }

    void train_ann(){
        double MSE;
        int epochs = entry_epochs.getText().getIntValue();
        for (int i = 0; i < epochs; i++){
            MSE = 0.0;
            MSE += sdrnn->bp({1,1,1,1,1,1,0},{1,0,0,0,0,0,0,0,0,0}); //0 pattern
            MSE += sdrnn->bp({0,1,1,0,0,0,0},{0,1,0,0,0,0,0,0,0,0}); //1 pattern
            MSE += sdrnn->bp({1,1,0,1,1,0,1},{0,0,1,0,0,0,0,0,0,0}); //2 pattern
            MSE += sdrnn->bp({1,1,1,1,0,0,1},{0,0,0,1,0,0,0,0,0,0}); //3 pattern
            MSE += sdrnn->bp({0,1,1,0,0,1,1},{0,0,0,0,1,0,0,0,0,0}); //4 pattern
            MSE += sdrnn->bp({1,0,1,1,0,1,1},{0,0,0,0,0,1,0,0,0,0}); //5 pattern
            MSE += sdrnn->bp({1,0,1,1,1,1,1},{0,0,0,0,0,0,1,0,0,0}); //6 pattern
            MSE += sdrnn->bp({1,1,1,0,0,0,0},{0,0,0,0,0,0,0,1,0,0}); //7 pattern
            MSE += sdrnn->bp({1,1,1,1,1,1,1},{0,0,0,0,0,0,0,0,1,0}); //8 pattern
            MSE += sdrnn->bp({1,1,1,1,0,1,1},{0,0,0,0,0,0,0,0,0,1}); //9 pattern
        }
        MSE /= 10.0;
        lbl_err.setText(std::to_string(MSE), no);
        tepochs += epochs;
        lbl_tepochs.setText(std::to_string(tepochs), no);
        run_ann();
    }

    void buttonClicked(juce::Button* button) override{
        if (button == &btn_train)
            train_ann();
        if (button == &btn_reset){
            delete(sdrnn);
            sdrnn = new MultiLayerPerceptron({7,7,10});
            tepochs = 0;
            lbl_err.setText("---", no);
            lbl_tepochs.setText(std::to_string(tepochs), no);
            run_ann();
        }
    }

    void paint(juce::Graphics& g) override {
        g.fillAll(juce::Colour(240,240,240));
        
        g.setColour(juce::Colour(
            (int)(slider_a.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_a.getValue()* OFFSET),
            OFFSET - (int)(slider_a.getValue()* OFFSET)     ));
        g.fillRect(aX,aY,SEGL,SEGW);

        g.setColour(juce::Colour(
            (int)(slider_b.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_b.getValue()* OFFSET),
            OFFSET - (int)(slider_b.getValue()* OFFSET)     ));
        g.fillRect(bX,bY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_c.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_c.getValue()* OFFSET),
            OFFSET - (int)(slider_c.getValue()* OFFSET)     ));
        g.fillRect(cX,cY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_d.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_d.getValue()* OFFSET),
            OFFSET - (int)(slider_d.getValue()* OFFSET)     ));
        g.fillRect(dX,dY,SEGL,SEGW);

        g.setColour(juce::Colour(
            (int)(slider_e.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_e.getValue()* OFFSET),
            OFFSET - (int)(slider_e.getValue()* OFFSET)     ));
        g.fillRect(eX,eY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_f.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_f.getValue()* OFFSET),
            OFFSET - (int)(slider_f.getValue()* OFFSET)     ));
        g.fillRect(fX,fY,SEGW,SEGL);

        g.setColour(juce::Colour(
            (int)(slider_g.getValue()*(255-OFFSET)) + OFFSET,
            OFFSET - (int)(slider_g.getValue()* OFFSET),
            OFFSET - (int)(slider_g.getValue()* OFFSET)     ));
        g.fillRect(gX,gY,SEGL,SEGW);

    }

    MultiLayerPerceptron * sdrnn;
    juce::NotificationType no = juce::dontSendNotification;
    int tepochs = 0;

private:
    juce::Slider slider_a;
    juce::Slider slider_b;
    juce::Slider slider_c;
    juce::Slider slider_d;
    juce::Slider slider_e;
    juce::Slider slider_f;
    juce::Slider slider_g;

    juce::Label lbl_epochs_txt;
    juce::TextEditor entry_epochs;
    juce::TextButton btn_train;
    juce::Label lbl_err_txt; 
    juce::Label lbl_err;
    juce::Label lbl_tepochs_txt;
    juce::Label lbl_tepochs;
    juce::Label lbl_out0_txt;
    juce::Label lbl_out0;
    juce::Label lbl_out1_txt;
    juce::Label lbl_out1;
    juce::Label lbl_out2_txt;
    juce::Label lbl_out2;
    juce::Label lbl_out3_txt;
    juce::Label lbl_out3;
    juce::Label lbl_out4_txt;
    juce::Label lbl_out4;
    juce::Label lbl_out5_txt;
    juce::Label lbl_out5;
    juce::Label lbl_out6_txt;
    juce::Label lbl_out6;
    juce::Label lbl_out7_txt;
    juce::Label lbl_out7;
    juce::Label lbl_out8_txt;
    juce::Label lbl_out8;
    juce::Label lbl_out9_txt;
    juce::Label lbl_out9;
    juce::Label lbl_int_txt;
    juce::Label lbl_int;
    juce::TextButton btn_reset;
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MainContentComponent)
};
