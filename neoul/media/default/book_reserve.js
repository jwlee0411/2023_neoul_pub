$(document).ready(function() {
            $("#main03").hide();
            //    $("#main03").hide();

            $('input[type=checkbox][name=test_list]').change(function() {
                if ($(this).is(':checked')) {
                    //            alert(`${this.value} is checked`);
                    $("#main03").show();
                    //                $("#main03").show();
                } else {
                    //            alert(`${this.value} is unchecked`);
                    $("#main03").hide();
                    //                $("#main03").hide();
                }
            });


            $.ajax({
                url: 'https://cdn.discordapp.com/attachments/997106626999898174/1002149267605045279/csv_data.csv',
                dataType: 'text',
            }).done(successFunction);


            $("#button01").click(function(e) {
                alert("1 클릭");

            });



            $("#button02").click(function(e) {
                alert("2 클릭");

            });

            const minDate = getMinDate();
        const dateString = minDate.toLocaleDateString(); // 날짜를 원하는 로케일에 맞게 문자열로 변환
        const timeString = minDate.toLocaleTimeString(); // 시간을 원하는 로케일에 맞게 문자열로 변환
        const dateTimeString = `${dateString} ${timeString}`; // 날짜와 시간 문자열을 조합
        $('.timeStart').text = dateTimeString;
        console.log(dateTimeString);

        });





        $(function() {
            $('.datepicker').datepicker('setDate', '22-10-01');
        })

        $.datepicker.setDefaults({
            dateFormat: 'yy-mm-dd',
            //            prevText: '이전 달',
            //            nextText: '다음 달',
            monthNames: ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'],
            monthNamesShort: ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'],
            dayNames: ['일', '월', '화', '수', '목', '금', '토'],
            dayNamesShort: ['일', '월', '화', '수', '목', '금', '토'],
            dayNamesMin: ['일', '월', '화', '수', '목', '금', '토'],
            showMonthAfterYear: true,
            yearSuffix: '년',
            showOtherMonths: false,
            selectOtherMonths: false,
            minDate: getMinDate(),
            maxDate: getMaxDate(),

            showButtonPanel: false,


        });

        $(function() {
            $('.datepicker').datepicker();
        });


        function getMinDate()
        {
            var date = new Date();
            var minDate = new Date();
            minDate.setFullYear(date.getFullYear());
            minDate.setMonth(date.getMonth()+1);
            minDate.setDate(1);
            console.log(minDate);
            return minDate;
        }

        function getMaxDate()
        {
              var date = new Date();
            var minDate = new Date();
            minDate.setFullYear(date.getFullYear());
            minDate.setMonth(date.getMonth()+2);
            minDate.setDate(0);
            return minDate;
        }

        function successFunction(data) {
            alert(data);
        }